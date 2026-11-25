# AI Detection Lab — Autonomous Survey

Mission flow:
1. Load the student's trained YOLO `.pt` model from `config.py`.
2. Set camera to DOWN 90° before flight.
3. Take off and climb to the configured survey altitude.
4. Fly a lawnmower grid using **FORWARD + planner-controlled 90° turns**.
5. Run YOLO continuously on the live RTP camera stream.
6. When a new target is found, capture the annotated image and save the drone survey position.
7. Deduplicate repeated detections of the same target using a configurable spatial radius.
8. Continue until the complete survey plan is finished, then land.

The saved `captures/mission_*/detections.json` contains the captured image names, detections, waypoint, X/Y survey position, heading, and timestamp.

Important: the stored X/Y is the drone's survey position at detection time. Exact ground-object coordinates require camera calibration / footprint geometry and are not inferred from the YOLO bounding box in this version.
