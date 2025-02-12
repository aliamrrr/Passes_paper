import numpy as np
import supervision as sv
from utils.annotators import draw_pitch, draw_points_on_pitch, draw_pitch_voronoi_diagram, draw_paths_and_hull_on_pitch,draw_arrow_on_pitch
from utils.soccer import SoccerPitchConfiguration
from utils.view import ViewTransformer
import csv
import cv2
import numpy as np


CONFIG = SoccerPitchConfiguration()


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)


import supervision as sv
import cv2
from ultralytics import RTDETR
import numpy as np
from utils.models import load_player_detection_model,load_field_detection_model

PLAYER_ID = 0
GOALKEEPER_ID = 1
BALL_ID = 2
REFEREE_ID = 3
SIDE_REFEREE_ID = 4
STAFF_MEMBER_ID = 5

red = sv.Color.from_hex('FF0000')
yellow = sv.Color.from_hex('FFFF00')

model = load_player_detection_model(model_path="models/player_detect.pt")
keypoints_model = load_field_detection_model()

def process_frame(frame, team_classifier):
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    referees_detections = detections[detections.class_id == REFEREE_ID]
    side_referees_detections = detections[detections.class_id == SIDE_REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]
    staff_members_detections = detections[detections.class_id == STAFF_MEMBER_ID]
    players_detections = detections[detections.class_id == PLAYER_ID]
    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections
    )

    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])
    annotated_frame = frame.copy()


    class_colors = {
        "referee": (0, 255, 255), 
        "side_referee": (128, 0, 128),
        "ball": (255, 0, 0),
        "staff_member": (255, 255, 153)}

    for i in range(len(referees_detections)):
        x1, y1, x2, y2 = map(int, referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["referee"], 2)
        label = f"Referee ({referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["referee"], 2)

    for i in range(len(side_referees_detections)):
        x1, y1, x2, y2 = map(int, side_referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["side_referee"], 2)
        label = f"Side Referee ({side_referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["side_referee"], 2)

    for i in range(len(ball_detections)):
        x1, y1, x2, y2 = map(int, ball_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["ball"], 2)
        label = f"Ball ({ball_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["ball"], 2)

    for i in range(len(staff_members_detections)):
        x1, y1, x2, y2 = map(int, staff_members_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["staff_member"], 2)
        label = f"Staff Member ({staff_members_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["staff_member"], 2)

    for i in range(len(all_detections)):
        x1, y1, x2, y2 = map(int, all_detections.xyxy[i])
        team_color = (0, 255, 0) if all_detections.class_id[i] == 1 else (255, 0, 0)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), team_color, 2)
        label = f"{'Team 1' if all_detections.class_id[i] == 1 else 'Team 0'}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

    return annotated_frame


def draw_radar_view(frame, CONFIG, team_classifier, type='tactical'):

    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    players_detections = detections[detections.class_id == PLAYER_ID]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    annotated_frame = draw_pitch(CONFIG)

    if type == "tactical":
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.from_hex('FFD700'),
            edge_color=sv.Color.BLACK,
            radius=10,
            thickness=2,
            pitch=annotated_frame)

        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 0],
            face_color=sv.Color.from_hex('FFFF00'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
        
        annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 1],
            face_color=sv.Color.from_hex('FF0000'),
            edge_color=sv.Color.WHITE,
            radius=16,
            thickness=1,
            pitch=annotated_frame)
    
    elif type == "voronoi":
        print('we want voronoi!')
        annotated_frame = draw_pitch_voronoi_diagram(
            config=CONFIG,
            team_1_xy=pitch_players_xy[players_detections.class_id == 0],
            team_2_xy=pitch_players_xy[players_detections.class_id == 1],
            team_1_color=yellow,
            team_2_color=red,
            pitch=annotated_frame)

    return annotated_frame


def passes_options(frame, CONFIG, team_classifier, passes_mode):
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    players_detections = detections[detections.class_id == PLAYER_ID]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]
    referees_detections = detections[detections.class_id == REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]

    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])

    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    annotated_frame = draw_pitch(CONFIG)

    if passes_mode == 'build':
        print('mode 1')
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            distances = np.linalg.norm(pitch_players_xy - pitch_ball_xy, axis=1)
            closest_player_idx = np.argmin(distances)
            closest_player_team = players_detections.class_id[closest_player_idx]

            teammates_idx = np.where(players_detections.class_id == closest_player_team)[0]
            teammates_xy = pitch_players_xy[teammates_idx]

            opponent_team = 1 - closest_player_team
            opponent_idx = np.where(players_detections.class_id == opponent_team)[0]
            opponent_xy = pitch_players_xy[opponent_idx]

            pressure_radius = 500
            pressures = []

            for teammate in teammates_xy:
                pressure = sum(np.linalg.norm(teammate - opponent) < pressure_radius for opponent in opponent_xy)
                pressures.append(pressure)

            for i, teammate in enumerate(teammates_xy):
                if np.array_equal(teammate, pitch_ball_xy[0]):
                    continue

                distance = np.linalg.norm(teammate - pitch_ball_xy[0])
                pressure_factor = pressures[i]

                difficulty = np.clip(
                    1 - np.exp(-distance / 3000) * (1 + np.exp(pressure_factor)) * (1 - 0.5 * pressure_factor),
                    0,
                    1
                )


                print('distance', distance)
                print('pressure_factor', pressure_factor)
                print('difficulty', difficulty)

                color = sv.Color(
                    int(difficulty * 255),
                    int((1 - difficulty) * 255),  # Red component: dimmer as difficulty increases
                    0  # Blue component: always 0
                )

                annotated_frame = draw_arrow_on_pitch(
                    config=CONFIG,
                    xy_start=[pitch_ball_xy[0]],
                    xy_end=[teammate],
                    color=color,
                    thickness=7,
                    pitch=annotated_frame
                )

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.from_hex('FFD700'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 0],
                    face_color=sv.Color.from_hex('00BFFF'),
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 1],
                    face_color=sv.Color.from_hex('FF1493'),
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

    elif passes_mode == 'interception':
        print('mode 2')
        if len(ball_detections) > 0:
            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            distances = np.linalg.norm(pitch_players_xy - pitch_ball_xy, axis=1)
            closest_player_idx = np.argmin(distances)
            closest_player_team = players_detections.class_id[closest_player_idx]

            teammates_idx = np.where(players_detections.class_id == closest_player_team)[0]
            teammates_xy = pitch_players_xy[teammates_idx]

            opponent_team = 1 - closest_player_team
            opponent_idx = np.where(players_detections.class_id == opponent_team)[0]
            opponent_xy = pitch_players_xy[opponent_idx]

            pressure_radius = 500
            pressures = []

            for teammate in teammates_xy:
                pressure = sum(np.linalg.norm(teammate - opponent) < pressure_radius for opponent in opponent_xy)
                pressures.append(pressure)

            for i, teammate in enumerate(teammates_xy):
                if np.array_equal(teammate, pitch_ball_xy[0]):
                    continue

                distance = np.linalg.norm(teammate - pitch_ball_xy[0])
                pressure_factor = pressures[i]

                ball_to_teammate_vector = teammate - pitch_ball_xy[0]
                ball_to_teammate_vector_normalized = ball_to_teammate_vector / np.linalg.norm(ball_to_teammate_vector)

                interception_radius = 500
                interception_risk = 0

                for opponent in opponent_xy:
                    projection = np.dot(opponent - pitch_ball_xy[0], ball_to_teammate_vector_normalized)
                    projected_point = pitch_ball_xy[0] + projection * ball_to_teammate_vector_normalized

                    if np.linalg.norm(opponent - projected_point) < interception_radius:
                        interception_risk += 1

                    difficulty = np.clip(
                        (np.exp(-distance / 4000) * (1 + np.exp(pressure_factor)) * (1 - 0.5 * pressure_factor)) 
                        * (1 + (distance / 4000))
                        * (1 - (interception_risk * 0.5)),
                        0,
                        1
                    )

                    difficulty = 1 - difficulty

                    print('sidtance  '+ str(distance) + 'risk  ' + str(interception_risk) + 'diff  ' + str(difficulty))

                color = sv.Color(
                    int(difficulty * 255),
                    int((1 - difficulty) * 255),
                    0
                )

                annotated_frame = draw_arrow_on_pitch(
                    config=CONFIG,
                    xy_start=[pitch_ball_xy[0]],
                    xy_end=[teammate],
                    color=color,
                    thickness=7,
                    pitch=annotated_frame
                )

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.from_hex('FFD700'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    thickness=2,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 0],
                    face_color=yellow,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players_detections.class_id == 1],
                    face_color=red,
                    edge_color=sv.Color.WHITE,
                    radius=16,
                    thickness=1,
                    pitch=annotated_frame)

    return annotated_frame



def calculate_optimal_passes(frame, CONFIG, team_classifier, max_passes=3):
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    ball_detections = detections[detections.class_id == BALL_ID]

    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

    current_ball_pos = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
    all_players_xy = transformer.transform_points(players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
    team_ids = players_detections.class_id

    pitch_min_y = min(CONFIG.vertices, key=lambda p: p[1])[1]
    pitch_max_y = max(CONFIG.vertices, key=lambda p: p[1])[1]
    max_y_length = pitch_max_y - pitch_min_y

    distances = np.linalg.norm(all_players_xy - current_ball_pos, axis=1)
    closest_player_idx = np.argmin(distances)
    current_team = team_ids[closest_player_idx]

    pressure_radius = 500
    interception_radius = 500
    passes_sequence = []
    used_players = []

    for _ in range(max_passes):
        teammates_mask = (team_ids == current_team)
        available_mask = ~np.isin(np.arange(len(all_players_xy)), used_players)
        candidates_mask = teammates_mask & available_mask
        
        if not np.any(candidates_mask):
            break

        candidates_xy = all_players_xy[candidates_mask]
        opponents_xy = all_players_xy[team_ids != current_team]

        difficulties = []
        for candidate in candidates_xy:
            distance = np.linalg.norm(candidate - current_ball_pos)

            pressure = sum(np.linalg.norm(candidate - opponent) < pressure_radius for opponent in opponents_xy)

            pass_vector = candidate - current_ball_pos
            pass_vector_norm = pass_vector / np.linalg.norm(pass_vector)
            interceptions = 0
            
            for opponent in opponents_xy:
                projection = np.dot(opponent - current_ball_pos, pass_vector_norm)
                if projection < 0 or projection > np.linalg.norm(pass_vector):
                    continue
                closest_point = current_ball_pos + projection * pass_vector_norm
                if np.linalg.norm(opponent - closest_point) < interception_radius:
                    interceptions += 1

            difficulty = (distance / 4000) * (1 + pressure) * (1 + interceptions)

            delta_y = (candidate[0] - current_ball_pos[0])

            if delta_y>0 :
                difficulty = difficulty/1000
            else :
                difficulty = difficulty*100
    

            
            difficulties.append(difficulty)

        # best pass
        best_idx = np.argmin(difficulties)
        print('def')
        print(difficulties[best_idx])
        best_receiver = candidates_xy[best_idx]

        passes_sequence.append((current_ball_pos, best_receiver))
        used_players.append(np.where(candidates_mask)[0][best_idx])
        current_ball_pos = best_receiver

    annotated_frame = draw_pitch(CONFIG)

    for pass_start, pass_end in passes_sequence:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[pass_start],
            xy_end=[pass_end],
            color=sv.Color.GREEN,
            thickness=9,
            pitch=annotated_frame
        )

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=all_players_xy[team_ids == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=all_players_xy[team_ids == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.WHITE,
        radius=16,
        thickness=1,
        pitch=annotated_frame)

    return annotated_frame

def calculate_realistic_optimal_passes(frame, CONFIG, team_classifier, max_passes=2):

    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    players_detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    ball_detections = detections[detections.class_id == BALL_ID]


    result = keypoints_model.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)
    
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)

    current_ball_pos = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))[0]
    pitch_ball_pos = current_ball_pos.copy()
    all_players_xy = transformer.transform_points(players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
    team_ids = players_detections.class_id


    DEFENSIVE_PARAMS = {
        'reaction_radius': 800,
        'aggressivity': 30,
        'pass_duration': 1.2,
        'player_speed': 10.0
    }

    dynamic_opponents = all_players_xy[team_ids != team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]].copy()
    reaction_movements = []
    original_teammates = all_players_xy[team_ids == team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]].copy()

    passes_sequence = []
    used_players = []
    current_possession = team_ids[np.argmin(np.linalg.norm(all_players_xy - current_ball_pos, axis=1))]

    for pass_num in range(max_passes):
        teammates_mask = (team_ids == current_possession)
        available_mask = ~np.isin(np.arange(len(all_players_xy)), used_players)
        candidates_mask = teammates_mask & available_mask
        
        if not np.any(candidates_mask):
            break

        candidates_xy = all_players_xy[candidates_mask]
        opponents_xy = dynamic_opponents

        difficulties = []
        for candidate in candidates_xy:
            distance = np.linalg.norm(candidate - current_ball_pos)
  
            pressure = sum(np.linalg.norm(candidate - opp) < 500 for opp in opponents_xy)
            pass_vector = candidate - current_ball_pos
            pass_vector_norm = pass_vector / np.linalg.norm(pass_vector)
            interceptions = 0
            
            for opp in opponents_xy:
                projection = np.dot(opp - current_ball_pos, pass_vector_norm)
                if 0 < projection < np.linalg.norm(pass_vector):
                    closest_point = current_ball_pos + projection * pass_vector_norm
                    if np.linalg.norm(opp - closest_point) < 400:
                        interceptions += 1

            delta_x = candidate[0] - current_ball_pos[0]
            direction_bonus = 20 if delta_x < 0 else 0.5
            difficulty = (distance / 3500) * (1 + pressure) * (1 + interceptions) * direction_bonus
            difficulties.append(difficulty)

        best_idx = np.argmin(difficulties)
        best_receiver = candidates_xy[best_idx]
        passes_sequence.append((current_ball_pos, best_receiver))
        used_players.append(np.where(candidates_mask)[0][best_idx])

        if len(opponents_xy) > 0:
            max_move = DEFENSIVE_PARAMS['player_speed'] * DEFENSIVE_PARAMS['pass_duration']
            distances_to_receiver = np.linalg.norm(dynamic_opponents - best_receiver, axis=1)

            closest_defenders = np.argsort(distances_to_receiver)[:2]
            random_defender = np.random.choice(len(dynamic_opponents)) if len(dynamic_opponents) > 2 else None
            
            for def_idx in list(closest_defenders) + [random_defender]:
                if def_idx is None or def_idx >= len(dynamic_opponents):
                    continue
                
                defender_pos = dynamic_opponents[def_idx]
                direction = best_receiver - defender_pos
                move_dist = min(np.linalg.norm(direction), max_move)

                print((direction / np.linalg.norm(direction)) * move_dist * DEFENSIVE_PARAMS['aggressivity'])
                new_pos = defender_pos + (direction / np.linalg.norm(direction)) * move_dist * DEFENSIVE_PARAMS['aggressivity']
                new_pos = np.clip(new_pos, [0, 0], [12000, 7000])
                
                reaction_movements.append({
                    'start': defender_pos.copy(),
                    'end': new_pos.copy(),
                    'team': 'defense'
                })
                dynamic_opponents[def_idx] = new_pos

        current_ball_pos = best_receiver

    annotated_frame = draw_pitch(CONFIG)

    for movement in reaction_movements:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[movement['start']],
            xy_end=[movement['end']],
            color=sv.Color.RED,
            thickness=8,
            pitch=annotated_frame
        )

    for pass_start, pass_end in passes_sequence:
        annotated_frame = draw_arrow_on_pitch(
            config=CONFIG,
            xy_start=[pass_start],
            xy_end=[pass_end],
            color=sv.Color.GREEN,
            thickness=8,
            pitch=annotated_frame
        )

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=dynamic_opponents,
        face_color=yellow,
        edge_color=sv.Color.WHITE,
        radius=14,
        thickness=1,
        pitch=annotated_frame)

    annotated_frame = draw_points_on_pitch(
        config=CONFIG,
        xy=original_teammates,
        face_color=red,
        edge_color=sv.Color.WHITE,
        radius=14,
        thickness=1,
        pitch=annotated_frame)

    return annotated_frame