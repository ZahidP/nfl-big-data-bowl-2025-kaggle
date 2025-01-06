import ipywidgets as widgets
from IPython.display import display, HTML
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data


class RoutePredictor(widgets.VBox):
    def __init__(self):
        # Create route definitions
        self.route_definitions = {
            "Slant": {
                "path": lambda x, y: f"M {x} {y} l 40 -40",  # Diagonal up
                "color": "#ffeb3b",
            },
            "Go Route": {
                "path": lambda x, y: f"M {x} {y} l 0 -80",  # Straight up
                "color": "#ffeb3b",
            },
            "Out Route": {
                "path": lambda x, y: (
                    # If x position is in left half of field, route goes left
                    f"M {x} {y} l 0 -40 l -60 0"
                    if x < 300
                    else
                    # If x position is in right half of field, route goes right
                    f"M {x} {y} l 0 -40 l 60 0"
                ),
                "color": "#ffeb3b",
            },
            "In Route": {
                "path": lambda x, y: (
                    # If x position is in left half of field, route goes right
                    f"M {x} {y} l 0 -40 l 60 0"
                    if x < 300
                    else
                    # If x position is in right half of field, route goes left
                    f"M {x} {y} l 0 -40 l -60 0"
                ),
                "color": "#ffeb3b",
            },
        }

        # Create player controls dictionary with default positions
        self.players = {
            # Offensive players - positioned at bottom
            "WR1": {
                "color": "blue",
                "active": True,
                "player_id": 1001,
                "type": "offense",
                "default_x": 15,
                "default_y": 70,
            },  # Far left WR
            "WR2": {
                "color": "blue",
                "active": False,
                "player_id": 1002,
                "type": "offense",
                "default_x": 30,
                "default_y": 70,
            },  # Inside left WR
            "WR3": {
                "color": "blue",
                "active": False,
                "player_id": 1003,
                "type": "offense",
                "default_x": 70,
                "default_y": 70,
            },  # Inside right WR
            "WR4": {
                "color": "blue",
                "active": False,
                "player_id": 1004,
                "type": "offense",
                "default_x": 85,
                "default_y": 70,
            },  # Far right WR
            "TE": {
                "color": "purple",
                "active": False,
                "player_id": 1005,
                "type": "offense",
                "default_x": 40,
                "default_y": 70,
            },  # Tight end position
            # Defensive players - positioned at top
            "CB1": {
                "color": "red",
                "active": True,
                "player_id": 2001,
                "type": "defense",
                "default_x": 15,
                "default_y": 50,
            },  # Matching WR1
            "CB2": {
                "color": "red",
                "active": False,
                "player_id": 2002,
                "type": "defense",
                "default_x": 30,
                "default_y": 50,
            },  # Matching WR2
            "SS": {
                "color": "orange",
                "active": False,
                "player_id": 2003,
                "type": "defense",
                "default_x": 45,
                "default_y": 30,
            },  # Strong safety deeper
            "FS": {
                "color": "orange",
                "active": False,
                "player_id": 2004,
                "type": "defense",
                "default_x": 55,
                "default_y": 30,
            },  # Free safety deeper
        }

        # Initialize controls for each player
        for player_id in self.players:
            self.players[player_id].update(
                {
                    "active_checkbox": widgets.Checkbox(
                        value=self.players[player_id]["active"],
                        description=f"Show {player_id}",
                        style={"description_width": "initial"},
                    ),
                    "x_slider": widgets.FloatSlider(
                        value=self.players[player_id]["default_x"],
                        min=0,
                        max=100,
                        step=1,
                        description=f"{player_id} X:",
                        style={"description_width": "initial"},
                        disabled=not self.players[player_id]["active"],
                    ),
                    "y_slider": widgets.FloatSlider(
                        value=self.players[player_id]["default_y"],
                        min=0,
                        max=100,
                        step=1,
                        description=f"{player_id} Y:",
                        style={"description_width": "initial"},
                        disabled=not self.players[player_id]["active"],
                    ),
                }
            )

            # Bind checkbox event to enable/disable sliders
            self.players[player_id]["active_checkbox"].observe(
                lambda change, pid=player_id: self.toggle_player(change, pid),
                names="value",
            )

        # RB control
        self.rb_checkbox = widgets.Checkbox(
            value=False, description="Show RB", style={"description_width": "initial"}
        )

        self.calculate_button = widgets.Button(
            description="Calculate Routes", button_style="success"
        )

        # Create output area for predictions
        self.output = widgets.Output()

        # Create the field visualization
        self.field = widgets.HTML()

        # Update the field initially
        self.update_field()

        # Bind control events
        for player in self.players.values():
            player["x_slider"].observe(lambda _: self.update_field(), names="value")
            player["y_slider"].observe(lambda _: self.update_field(), names="value")
        self.rb_checkbox.observe(lambda _: self.update_field(), names="value")
        self.calculate_button.on_click(self.on_calculate)

        # Create controls layout with player groups
        controls = [widgets.HTML("<h3>Player Controls</h3>")]

        # Group players by type (offense/defense)
        offense_controls = [widgets.HTML("<h4>Offense</h4>")]
        defense_controls = [widgets.HTML("<h4>Defense</h4>")]

        for player_id, player in self.players.items():
            player_group = widgets.VBox(
                [
                    player["active_checkbox"],
                    player["x_slider"],
                    player["y_slider"],
                    widgets.HTML("<hr>"),  # Divider between players
                ]
            )
            if player["type"] == "offense":
                offense_controls.append(player_group)
            else:
                defense_controls.append(player_group)

        controls.extend(
            [
                widgets.VBox(offense_controls),
                widgets.VBox(defense_controls),
                self.rb_checkbox,
                self.calculate_button,
            ]
        )

        controls_box = widgets.VBox(controls)

        # Initialize the parent widget
        super().__init__([self.field, controls_box, self.output])

    def toggle_player(self, change, player_id):
        self.players[player_id]["active"] = change["new"]
        self.players[player_id]["x_slider"].disabled = not change["new"]
        self.players[player_id]["y_slider"].disabled = not change["new"]
        self.update_field()

    def update_field(self):
        # Create field HTML with yard lines and SVG overlay for routes
        yard_lines = ""
        for i in range(10):
            yard_lines += f"""
                <div style="position: absolute; 
                           left: {i * 10}%; 
                           height: 100%; 
                           width: 1px; 
                           background-color: white; 
                           opacity: 0.3;">
                </div>
            """

        # Create player dots and routes
        player_dots = ""
        routes_svg = ""

        # Calculate the SVG viewport dimensions based on the field size
        field_width = 600
        field_height = 300

        for player_id, player in self.players.items():
            if player["active_checkbox"].value:
                # Calculate actual pixel coordinates
                x_pixels = (player["x_slider"].value / 100) * field_width
                y_pixels = (player["y_slider"].value / 100) * field_height

                # Add player dot
                player_dots += f"""
                    <div style="position: absolute; 
                               left: {player['x_slider'].value}%; 
                               top: {player['y_slider'].value}%; 
                               width: 20px; 
                               height: 20px; 
                               background-color: {player['color']}; 
                               border-radius: 50%; 
                               transform: translate(-50%, -50%);">
                        <div style="position: absolute; 
                                  top: -20px; 
                                  left: 50%; 
                                  transform: translateX(-50%); 
                                  color: white; 
                                  font-size: 12px; 
                                  text-shadow: 1px 1px 1px black;">
                            {player_id}
                        </div>
                    </div>
                """

                # Add route if it's an offensive player
                if player["type"] == "offense":
                    routes_svg += f"""
                        <path d="{self.route_definitions['Slant']['path'](x_pixels, y_pixels)}"
                              stroke="{self.route_definitions['Slant']['color']}"
                              stroke-width="2"
                              fill="none"
                              opacity="0.6"/>
                    """

        # Add RB if enabled
        rb_dot = (
            """
            <div style="position: absolute; 
                       left: 45%; 
                       top: 70%; 
                       width: 20px; 
                       height: 20px; 
                       background-color: red; 
                       border-radius: 50%; 
                       transform: translate(-50%, -50%);">
                <div style="position: absolute; 
                           top: -20px; 
                           left: 50%; 
                           transform: translateX(-50%); 
                           color: white; 
                           font-size: 12px; 
                           text-shadow: 1px 1px 1px black;">
                    RB
                </div>
            </div>
        """
            if self.rb_checkbox.value
            else ""
        )

        # Create SVG overlay for routes
        svg_overlay = f"""
            <svg style="position: absolute; 
                       top: 0; 
                       left: 0; 
                       width: 100%; 
                       height: 100%; 
                       pointer-events: none;">
                {routes_svg}
            </svg>
        """

        self.field.value = f"""
            <div style="width: 600px; 
                       height: 300px; 
                       background-color: #4CAF50; 
                       margin: 10px; 
                       border: 2px solid black;
                       position: relative;
                       overflow: hidden;">
                {yard_lines}
                {player_dots}
                {rb_dot}
                {svg_overlay}
            </div>
        """

    def create_batch_data(self):
        """Creates a batch of data for route prediction based on current player positions"""
        active_players = [
            pid for pid, p in self.players.items() if p["active_checkbox"].value
        ]
        num_players = len(active_players)

        # Create a DataFrame with player positions and attributes
        player_data = []
        for player_id, player in self.players.items():
            if player["active_checkbox"].value:
                player_data.append(
                    {
                        "nflId": player["player_id"],
                        "position": player_id[
                            :2
                        ],  # Get position type (WR, TE, CB, SS, FS)
                        "x": player["x_slider"].value,
                        "y": player["y_slider"].value,
                        "weight": 200,  # Mock values
                        "height_inches": 72,
                        "s": 0,  # Speed set to 0 for static position
                        "is_offense": 1 if player["type"] == "offense" else 0,
                        "route_encoded": None,
                    }
                )

        frame_data = pd.DataFrame(player_data)

        # Create node features
        node_features = []
        player_ids = []
        eligible_mask = []

        for _, player in frame_data.iterrows():
            # Create position encoding
            position_tensor = torch.tensor([1 if player["position"] == "WR" else 2])
            weight_tensor = torch.tensor([player["weight"]])
            height_tensor = torch.tensor([player["height_inches"]])
            is_offense_tensor = torch.tensor([player["is_offense"]])
            motion_tensor = torch.tensor([np.abs(player["s"])])
            x_pos = torch.tensor([player["x"] / 100.0])  # Normalize to [0,1]
            y_pos = torch.tensor([player["y"] / 100.0])  # Normalize to [0,1]

            features = torch.cat(
                [
                    position_tensor,
                    weight_tensor,
                    height_tensor,
                    is_offense_tensor,
                    motion_tensor,
                    x_pos,
                    y_pos,
                ]
            )

            node_features.append(features)
            player_ids.append(player["nflId"])
            eligible_mask.append(
                1 if player["is_offense"] == 1 else 0
            )  # Only offensive players eligible

        # Create edge connections with distance and angle calculations
        edge_index = []
        edge_attr = []
        frame_idx = 0  # Single frame

        for i in range(len(frame_data)):
            for j in range(i + 1, len(frame_data)):
                node1 = frame_data.iloc[i]
                node2 = frame_data.iloc[j]

                # Calculate Euclidean distance
                dist = np.sqrt(
                    (node1["x"] - node2["x"]) ** 2 + (node1["y"] - node2["y"]) ** 2
                )

                if dist <= 100:  # Distance threshold
                    edge_index.extend([[i, j]])
                    edge_attr.extend(
                        [[dist, frame_idx, 1, 0]]
                    )  # Added 0 for potential additional feature

        # Convert everything to tensors with the correct shapes
        node_features = torch.stack(node_features)
        player_ids = torch.tensor(player_ids, dtype=torch.long)
        eligible_mask = torch.tensor(eligible_mask, dtype=torch.bool)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create time tensor (single value repeated for each node)
        time = torch.full((1,), 1800.0)  # 30 minutes in seconds

        # Create the Data object with the specified structure
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            time=time,
            play_id=torch.full_like(player_ids, 1),
            game_id=torch.full_like(player_ids, 1),
            batch=torch.zeros_like(player_ids, dtype=torch.long),
            eligible_mask=eligible_mask,
            route_targets=torch.zeros_like(player_ids, dtype=torch.long),
            down=torch.tensor([1]),
            distance=torch.tensor([10.0]),
            quarter=torch.tensor([1]),
            game_clock=torch.tensor([[1800.0]]),
            offense_team=torch.tensor([0]),
            offense_score=torch.tensor([0.0]),
            defense_score=torch.tensor([0.0]),
            week=torch.tensor([1]),
            yards_to_go=torch.tensor([10.0]),
            player_ids=player_ids,
            plays_elapsed=torch.zeros_like(player_ids, dtype=torch.long),
        )

        return data

    def predict_routes(self, data):
        """
        Mock route prediction based on player positions
        Returns probabilities for each route type
        """
        predictions = {}
        for i, player_id in enumerate(data.player_ids):
            if data.eligible_mask[i]:
                x_pos = data.x[i, -2].item() * 100  # Convert back to 0-100 scale
                y_pos = data.x[i, -1].item() * 100

                # Base probabilities
                slant_prob = 40
                go_route_prob = 30
                out_route_prob = 30

                # Adjust based on x position
                if x_pos > 70:
                    go_route_prob += 10
                    slant_prob -= 5
                    out_route_prob -= 5
                elif x_pos < 30:
                    out_route_prob += 10
                    slant_prob += 5
                    go_route_prob -= 15

                # Adjust based on y position
                if y_pos > 70:
                    out_route_prob += 15
                    slant_prob -= 10
                    go_route_prob -= 5

                # Find player ID in our players dict
                player_name = next(
                    name
                    for name, p in self.players.items()
                    if p["player_id"] == player_id.item()
                )

                predictions[player_name] = {
                    "slant": max(0, min(100, slant_prob)),
                    "go_route": max(0, min(100, go_route_prob)),
                    "out_route": max(0, min(100, out_route_prob)),
                }

        return predictions

    def on_calculate(self, b):
        with self.output:
            self.output.clear_output()

            # Create batch data from current positions
            batch_data = self.create_batch_data()

            # Get predictions
            predictions = self.predict_routes(batch_data)

            print("Predicted Routes:")
            for player_id, routes in predictions.items():
                print(f"\n{player_id}:")
                print(f"Slant: {routes['slant']}%")
                print(f"Go Route: {routes['go_route']}%")
                print(f"Out Route: {routes['out_route']}%")
