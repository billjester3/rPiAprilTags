from cscore import CameraServer
import ntcore
import numpy
import cv2
import robotpy_apriltag
import wpimath.units
from wpimath.geometry import Pose3d, Transform3d, Translation3d, Rotation3d, Quaternion, CoordinateSystem

# Config info for the pose estimator. This is dependent on the camera used
poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(
    0.1651,  #tag size in meters
    543.93,  #Fx: x focal length
    544.98,  #Fy: y focal length
    316.29,  #Cx: x focal center (based on 640x480 resolution)
    250.55,  #Cy: y focal center (based on 640x480 resolution)
)

# create the PoseEstimator
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)

# store the robot to camera transform
robotToCamera = Transform3d(Translation3d(0.2, 0.0, 1.0),
                            Rotation3d())

# read the field data into the AprilTagFieldLayout object
aprilTagFieldLayout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

# create NetworkTables server (change to client when this is on the robot)
ntInstance = ntcore.NetworkTableInstance.getDefault()
ntInstance.startServer()

# create some NetworkTable publishers
table = ntInstance.getTable("AprilTags")
tagIdPublisher = table.getIntegerTopic("Tag ID").publish()
tagFoundPublisher = table.getBooleanTopic("Tag Found?").publish()

poseTable = table.getSubTable("Robot Pose")
xPublisher = poseTable.getDoubleTopic("X").publish()
yPublisher = poseTable.getDoubleTopic("Y").publish()
zPublisher = poseTable.getDoubleTopic("Z").publish()


#Camera constants
xResolution = 640
yResolution = 480
frameRate = 30

# start up CameraServer with a USB camera
camera = CameraServer.startAutomaticCapture()
CameraServer.enableLogging()
camera.setResolution(xResolution, yResolution)
camera.setFPS(frameRate)

# create output sync for modified image
cvSink = CameraServer.getVideo()
outputStream = CameraServer.putVideo("Vision", xResolution, yResolution)

# pre-create color and black and white OpenCV images
mat = numpy.zeros(shape=(xResolution, yResolution, 3), dtype=numpy.uint8)
grayMat = numpy.zeros(shape=(xResolution, yResolution), dtype=numpy.uint8)

# color for tag outline
lineColor = (0,255,0)


# main loop
while True:
	# grab the latest frame from the camera
	_, mat = cvSink.grabFrame(mat)

	# convert to grayscale for AprilTag detection
	grayMat = cv2.cvtColor(mat, cv2.COLOR_RGB2GRAY)

	# run the AprilTagDetector
	detections = aprilTagDetector.detect(grayMat)

	if (detections != []):
        # Tags were found. estimate the transform from the camera to the (first) tag
		cameraToTag = poseEstimator.estimate(detections[0])
		tagId = detections[0].getId()
	else:
        # No tags found, so just store an empty transform
		cameraToTag = Transform3d()
		cameraPose = Pose3d()
		robotPose = Pose3d()
		tagId = 0

	# Outline all tags found
	for detection in detections:
		for i in range(4):
			j = (i + 1) % 4
			p1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
			p2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
			mat = cv2.line(mat, p1, p2, lineColor, 2)

	# Process the first tag
	if tagId > 0:
		# look up the pose for the tag in view
		tagPose = aprilTagFieldLayout.getTagPose(tagId)

		if tagPose is not None:
			# first we need to flip the Camera To Tag transform's angle 180 degrees around the y axis since the tag is oriented into the field
			flipTagRotation = Rotation3d(axis=(0,1,0), angle=wpimath.units.degreesToRadians(180))
			cameraToTag = Transform3d(cameraToTag.translation(), cameraToTag.rotation().rotateBy(flipTagRotation))

			# The Camera To Tag transform is in a East/Down/North coordinate system, but we want it in the WPILib standard North/West/Up
			cameraToTag = CoordinateSystem.convert(cameraToTag, CoordinateSystem.EDN(), CoordinateSystem.NWU())

			# We now have a corrected transform from the camera to the tag. Apply the inverse transform to the tag pose to get the camera's pose
			cameraPose = tagPose.transformBy(cameraToTag.inverse())

			# compute robot pose from robot to camera transform
			robotPose = cameraPose.transformBy(robotToCamera.inverse())

	# Update output stream
	outputStream.putFrame(mat)

	# update NetworkTables
	xPublisher.set(robotPose.x)
	yPublisher.set(robotPose.y)
	zPublisher.set(robotPose.z)
	tagIdPublisher.set(tagId)
	tagFoundPublisher.set(tagId > 0)
