####################
# osr_msgs abnormal message
####################
# Abnormal ID
uint8 agent_id

# Report ID
int64 report_id # local time without nsec (python: time.time() or rostime.now() )

# Abnormal Status (0: NORMAL, 1: CAUTION, 2: ALERT)
uint8 status

# Abnormal Type (1: Map-based Unfamiliar Abnormal, 2: Map-based User-defined Abnormal, 3: Unfamiliar Object, 4: Unfamiliar Pedestrian, 5: Abnormal Sound) 
uint8 type

# Abnormal Detail
# FOR Map-based Unfamiliar Abnormal (type = 1), 1: Human Placement, 2: Car Placement, 3: Elevation Distribution, 4: Thermal Distribution, 5: Human Trajectory
# FOR Map-based User-defined Abnormal (type = 2), 1: Human Placement, 2: Car Placement, 3: Elevation Distribution, 4: Thermal Distribution
# FOR Unfamiliar Object (type = 3), 6: Unfamiliar Object
# FOR Unfamiliar Pedestrian (type = 4), 7: Unfamiliar Pedestrian
# FOR Abnormal Sound (type = 5), 8: Scream, 10: Explosion, 3: Unknown
uint8 detail

# Map Position
float64 pos_x
float64 pos_y

# Abnormal Score
uint8 score
