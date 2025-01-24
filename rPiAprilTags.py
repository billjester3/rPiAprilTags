from cscore import CameraServer
import ntcore
import numpy
import cv2
import robotpy_apriltag
import json
import wpimath.units
from wpimath.geometry import Pose3d, Transform3d, Translation3d, Rotation3d, Quaternion, CoordinateSystem

# Config info for the pose estimator. This is dependent on the camera used
tagSize = 0.1651  #tag size in meters
Fx = 671.23  #x focal length
Fy = 675.13  #y focal length
Cx = 329.84  #x focal center (based on 640x480 resolution)
Cy = 223.34  #y focal center (based on 640x480 resolution)

poseEstimatorConfig = robotpy_apriltag.AprilTagPoseEstimator.Config(tagSize, Fx, Fy, Cx, Cy)

# Pull the camera distortion values off the PhotonVision OpenCV calibration
cameraDistortion = numpy.float32([0.175, -1.194, -0.008, 0.001, 2.25 ])
cameraIntrinsics = numpy.eye(3)
cameraIntrinsics[0][0] = Fx
cameraIntrinsics[1][1] = Fy
cameraIntrinsics[0][2] = Cx
cameraIntrinsics[1][2] = Cy

# create the PoseEstimator
poseEstimator = robotpy_apriltag.AprilTagPoseEstimator(poseEstimatorConfig)

# store the robot to camera transform
robotToCamera = Transform3d(Translation3d(0.0, 0.0, 0.0),
                            Rotation3d())

# read the field data into the AprilTagFieldLayout object
aprilTagFieldLayout = robotpy_apriltag.AprilTagFieldLayout("TagPoses.json")

aprilTagDetector = robotpy_apriltag.AprilTagDetector()
aprilTagDetector.addFamily("tag36h11", 3)

# set configuration for the AprilTagDetector
# see https://robotpy.readthedocs.io/projects/robotpy/en/latest/robotpy_apriltag/AprilTagDetector.html for options
aprilTagDetectorConfig = aprilTagDetector.getConfig()
aprilTagDetectorConfig.numThreads = 4
aprilTagDetectorConfig.debug = 0
aprilTagDetectorConfig.refineEdges = True
aprilTagDetectorConfig.quadSigma = 0.5
aprilTagDetectorConfig.quadDecimate = 1.0
aprilTagDetector.setConfig(aprilTagDetectorConfig)

quadThresholdParameters = aprilTagDetector.getQuadThresholdParameters()
quadThresholdParameters.minClusterPixels = 5
quadThresholdParameters.maxNumMaxima = 10
quadThresholdParameters.criticalAngle = 0.79 # 45 degrees
quadThresholdParameters.maxLineFitMSE = 10.0
quadThresholdParameters.minWhiteBlackDiff = 5
quadThresholdParameters.deglitch = False
aprilTagDetector.setQuadThresholdParameters(quadThresholdParameters)


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


# start up CameraServer with a USB camera
camera = CameraServer.startAutomaticCapture()
CameraServer.enableLogging()

# read camera json settings.
# for a list of valid settings/ranges on the camera, run:
#   v4l2-ctl --list-ctrls --device /dev/video0
cameraConfig = open("cameraConfig.json", "r")
cameraConfigJson = json.load(cameraConfig)
camera.setConfigJson(cameraConfigJson)

xResolution = camera.getVideoMode().width
yResolution = camera.getVideoMode().height

print("Camera resolution: (", xResolution, ",", yResolution, ")")

# create output sync for modified image
cvSink = CameraServer.getVideo()
outputStream = CameraServer.putVideo("Vision", xResolution, yResolution)

# pre-create color and black and white OpenCV images
mat = numpy.zeros(shape=(xResolution, yResolution, 3), dtype=numpy.uint8)
grayMat = numpy.zeros(shape=(xResolution, yResolution), dtype=numpy.uint8)

# color for tag outline
lineColor = (0,255,0)
lineColor2 = (0, 0, 255)

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
		detection = detections[0]

		# Undistort the corners using the camera calibration
		corners = list(detection.getCorners(numpy.empty(8)))

		# Outline the tag using original corners
		for i in range(4):
			j = (i + 1) % 4
			p1 = (int(corners[2*i]),int(corners[2*i+1]))
			p2 = (int(corners[2*j]),int(corners[2*j+1]))
			mat = cv2.line(mat, p1, p2, lineColor, 2)

		distortedCorners = numpy.empty([4,2], dtype=numpy.float32)
		for i in range(4):
			distortedCorners[i][0]=corners[2*i]
			distortedCorners[i][1]=corners[2*i+1]

		# run the OpenCV undistortion routine to fix the corners
		undistortedCorners = cv2.undistortImagePoints(distortedCorners, cameraIntrinsics, cameraDistortion)
		for i in range(4):
			corners[2*i] = undistortedCorners[i][0][0]
			corners[2*i+1] = undistortedCorners[i][0][1]

		# Overlay undistorted corners too
		for i in range(4):
			j = (i + 1) % 4
			p1 = (int(corners[2*i]),int(corners[2*i+1]))
			p2 = (int(corners[2*j]),int(corners[2*j+1]))
			mat = cv2.line(mat, p1, p2, lineColor2, 2)

		# run the pose estimator using the fixed corners
		cameraToTag = poseEstimator.estimate(
			homography=detection.getHomography(),
			corners = tuple(corners))
		tagId = detection.getId()
	else:
        # No tags found, so just store an empty transform
		cameraToTag = Transform3d()
		cameraPose = Pose3d()
		robotPose = Pose3d()
		tagId = 0

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
	xPublisher.set(round(robotPose.x,3))
	yPublisher.set(round(robotPose.y,3))
	zPublisher.set(round(robotPose.z,3))
	tagIdPublisher.set(tagId)
	tagFoundPublisher.set(tagId > 0)
