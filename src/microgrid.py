import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

class DualBusMicrogrid:
    def __init__(self, config):
        self.cfg = config
        self.graph = nx.Graph()
        self.nodes = [] 
        self.edges = []
        self._build_topology()
        self.time_step = 0 # Track simulation time (0-23 hours)
        
    def _build_topology(self):
        """
        Constructs the Hierarchical Dual-Bus Graph.
        Level 0: Main Bus (Root)
        Level 1: Local Station Buses
        Level 2: EV Chargers (Loads)
        """
        # 1. Root Node (Main Bus / PCC)
        # ID 0 is always the Slack Bus (Voltage Source)
        self.graph.add_node(0, type='root', voltage=self.cfg['v_source'], power=0.0)
        
        node_counter = 1
        station_nodes = []
        
        for s in range(self.cfg['num_stations']):
            # 2. Local Station Bus
            station_id = node_counter
            self.graph.add_node(station_id, type='station', voltage=1.0, power=0.0)
            
            # Feeder Line (Main -> Station)
            # Switchable edge (status=1 means closed/connected)
            self.graph.add_edge(0, station_id, r=self.cfg['r_feeder'], status=1, type='feeder')
            station_nodes.append(station_id)
            node_counter += 1
            
            # 3. Components (Chargers, PV, Storage)
            # A. EV Chargers (Loads)
            for c in range(self.cfg['chargers_per_station']):
                charger_id = node_counter
                self.graph.add_node(charger_id, type='charger', voltage=1.0, power=0.0)
                self.graph.add_edge(station_id, charger_id, r=self.cfg['r_branch'], status=1, type='branch')
                node_counter += 1

            # B. PV Units (Generators)
            for p in range(self.cfg.get('pv_per_station', 0)):
                pv_id = node_counter
                self.graph.add_node(pv_id, type='pv', voltage=1.0, power=0.0)
                self.graph.add_edge(station_id, pv_id, r=self.cfg['r_branch'], status=1, type='branch')
                node_counter += 1

            # C. Storage Units (Batteries)
            for b in range(self.cfg.get('storage_per_station', 0)):
                batt_id = node_counter
                self.graph.add_node(batt_id, type='storage', voltage=1.0, power=0.0, soc=0.5)
                self.graph.add_edge(station_id, batt_id, r=self.cfg['r_branch'], status=1, type='branch')
                node_counter += 1
                
        # 4. Tie-Lines (Ring Topology)
        # Connect Station i to Station i+1 (Circular)
        for i in range(len(station_nodes)):
            u = station_nodes[i]
            v = station_nodes[(i + 1) % len(station_nodes)] # Wrap around
            self.graph.add_edge(u, v, r=self.cfg.get('r_tie', 0.03), status=0, type='tie')

        # Cache lists for faster indexing during solving
        self.nodes = list(self.graph.nodes(data=True))
        self.edges = list(self.graph.edges(data=True))

    def randomize_topology(self):
        """
        Randomly reconfigures switches to create diverse topologies for training.
        """
        # 1. Reset to Baseline (Radial)
        for u, v, d in self.graph.edges(data=True):
            if d['type'] == 'feeder': d['status'] = 1
            if d['type'] == 'tie': d['status'] = 0
            
        # 2. Random Reconfiguration (20% chance)
        if np.random.random() < 0.2:
            stations = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'station']
            if len(stations) < 2: return
            
            idx = np.random.randint(len(stations))
            s_target = stations[idx]
            s_neighbor = stations[(idx + 1) % len(stations)]
            
            # Open Feeder, Close Tie
            if self.graph.has_edge(0, s_target): self.graph[0][s_target]['status'] = 0
            if self.graph.has_edge(s_target, s_neighbor): self.graph[s_target][s_neighbor]['status'] = 1

    def update_stochastic_loads(self):
        """
        Simulates time-varying loads and generation (EV, PV, Battery).
        """
        # Advance time (simple 24h cycle loop)
        self.time_step = (self.time_step + 1) % 24
        hour = self.time_step

        # Solar Profile (Bell curve peaking at noon)
        # Peak generation ~ 0.8 p.u.
        solar_potential = max(0, 0.8 * np.sin((hour - 6) * np.pi / 12)) 
        # Add noise
        solar_potential *= np.random.uniform(0.8, 1.0)

        for node_id, data in self.graph.nodes(data=True):
            # 1. EV Chargers (Stochastic Load)
            if data['type'] == 'charger':
                # Higher probability of charging in evening (17:00 - 21:00)
                prob = 0.6 if 17 <= hour <= 21 else 0.2
                if np.random.random() < prob:
                    data['power'] = -np.random.uniform(0.0, self.cfg['load_scale'])
                else:
                    data['power'] = 0.0
            
            # 2. PV Units (Generation)
            elif data['type'] == 'pv':
                data['power'] = solar_potential

            # 3. Storage (Arbitrage)
            elif data['type'] == 'storage':
                # Randomize SOC (0.1 to 0.9) so the model learns from different battery states
                data['soc'] = np.random.uniform(0.1, 0.9)
                # Simple logic: Charge if solar is high, Discharge if evening
                if solar_potential > 0.5:
                    data['power'] = -0.2 # Charging
                elif 18 <= hour <= 22:
                    data['power'] = 0.3  # Discharging
                else:
                    data['power'] = 0.0

    def solve_power_flow(self):
        """
        Solves the DC Power Flow equation: Y * V = I
        Returns:
            state_vector: List of [Voltage, Power] for all nodes
            edge_vector: List of [Current, Status] for all edges
            total_loss: System-wide I^2*R loss
        """
        n = len(self.nodes)
        node_map = {node_id: i for i, (node_id, _) in enumerate(self.nodes)}
        
        # --- Step A: Build Y Matrix and I Vector ---
        rows, cols, data = [], [], []
        I_vec = np.zeros(n)
        
        # Fill Current Injections (I ~ P in DC approx)
        for i, (node_id, attr) in enumerate(self.nodes):
            I_vec[i] = attr['power']

        # Fill Admittance (Y)
        for u, v, attr in self.edges:
            if attr['status'] == 0: continue # Open switch = No conductance
            
            u_idx, v_idx = node_map[u], node_map[v]
            g = 1.0 / (attr['r'] + 1e-6)
            
            # Off-diagonal terms (-g)
            rows.extend([u_idx, v_idx])
            cols.extend([v_idx, u_idx])
            data.extend([-g, -g])
            
            # Diagonal terms (+g)
            rows.extend([u_idx, v_idx])
            cols.extend([u_idx, v_idx])
            data.extend([g, g])

        # Slack Bus Constraint (Node 0 fixed to V_source)
        # We add a massive conductance to ground at Node 0 to force V=V_source
        root_idx = node_map[0]
        rows.append(root_idx)
        cols.append(root_idx)
        data.append(1e6) 
        I_vec[root_idx] += 1e6 * self.cfg['v_source']

        # --- Step B: Solve Linear System ---
        Y = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
        solver = splu(Y)
        voltages = solver.solve(I_vec)
        
        # --- Step C: Compute Results ---
        total_loss = 0.0
        edge_results = []
        
        # Update Node Voltages in Graph
        node_results = []
        for i, (node_id, attr) in enumerate(self.nodes):
            attr['voltage'] = float(voltages[i])
            # Add SOC for storage nodes, default to 0 for others
            soc = attr.get('soc', 0.0)
            node_results.append({'id': node_id, 'v': attr['voltage'], 'p': attr['power'], 'type': attr['type'], 'soc': soc})

        # Calculate Edge Currents
        for u, v, attr in self.edges:
            u_idx, v_idx = node_map[u], node_map[v]
            
            if attr['status'] == 1:
                v_diff = voltages[u_idx] - voltages[v_idx]
                current = abs(v_diff / attr['r'])
                loss = (current ** 2) * attr['r']
                total_loss += loss
            else:
                current = 0.0
            
            edge_results.append({'u': u, 'v': v, 'current': float(current), 'status': attr['status'], 'r': attr['r']})

        return node_results, edge_results, total_loss
