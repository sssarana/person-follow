# Speed calculation depending on the distance to the object
When there is only one camera available, or if the light weight model is desirable, we want to estimate the distance as simply as possible.
In this case, we need to know:
1. The approximate distance to the person to adjust the speed of the autonomous car.
2. The approximate location of the person in the frame to adjust the steering angle.

All this can be achieved using bounding box coordinates.

## Object detection
For object detection, it is preferable to use a lightweight model that is only trained to recognise one specific object.
> I am using pretrained YOLOv5s, but it is worth retraining it to recognise people only. For now, COCO dataset is fine.

It is important to note that classification models will not do the job, since the most important part for us is the bounding box coordinates.
**YOLO results look like this:**
|   | xmin   | ymin  | xmax  | ymax  | confidence | class | name   |
|---|--------|-------|-------|-------|------------|-------|--------|
| 0 | 749.50 | 43.50 | 1148.0| 704.5 | 0.874023   | 0     | person |

Now we need to extract first 4 values for future calculations.
> You can use eiher class number - 0 for person, or class name.
## Extracting coordinates and finding area
The following function extracts the coordinates and save them in variables x1, x2, y1, y2.
From this we can find a center and calculate the area, which is also implemented below.
```python
 def process_detections(self, results):
        self.person_detected = False
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0:  # Class 0 is 'person' 
                x1, y1, x2, y2 = map(int, xyxy)
                self.person_center_x = (x1 + x2) // 2
                self.person_bbox_area = (x2 - x1) * (y2 - y1)
                self.person_detected = True
                self.get_logger().info(f"Person detected at center x: {self.person_center_x}, bbox area: {self.person_bbox_area}")
                break

        if self.person_detected:
            self.follow_person()
```
## Formulas
1. Speed calculation
**speed = max_speed - (self.person_bbox_area / desired_bbox_area) * (max_speed - min_speed)**
Where:
- desired_bbox_area is a reference area that defines when the person is considered "close"
- max_speed is the maximum speed of your choice and depending on your format, in our case it is 2 (m/s) 
- min_speed is the minimum speed of your choice and depending on your format, in our case it is 0.5 (m/s)

2. Steering angle calculation
The steering angle is calculated based on the error between the person's center x-coordinate and the frame center x-coordinate. 
The error is then processed through a PID controller to get the desired steering angle.
```python
 def pid_control_steering(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        self.get_logger().debug(f"PID control steering output: {output}")
        return output
```
Where:
- error is the difference between the person's center x-coordinate and the frame center x-coordinate
- kp is the proportional gain
- kd is the derivative gain
- ki is the integral gain

Then we need to call the function adjust the output , this is done in the person following function using the lines below.
```python
frame_center_x = 320 # because the frame size is 640
error = self.person_center_x - frame_center_x
self.get_logger().info(f"Following person, error: {error}")
steering_correction = self.pid_control_steering(error / frame_center_x)
steering_angle = np.clip(steering_correction, -np.pi/6, np.pi/6)
```
