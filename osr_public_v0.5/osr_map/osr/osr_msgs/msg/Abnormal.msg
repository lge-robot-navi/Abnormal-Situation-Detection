####################
# osr_msgs abnormal message
####################
# agent id
int8 agent_id

# Status (0: NORMAL, 1: CAUTION, 2: ALERT)
int8 status

# Timestamp
uint64 timestamp # uint64_t

# Abnormal Type (1: HUMAN, 2: CAR, 3: ELEVATION, 4: THERMAL, 5: SOUND) 
int8 abnormal_type

# Abnormal Detail
# FOR HUMAN (type = 1), 1: FALLEN, 2: INVASION, 3: ATYPICAL
# FOR CAR (type = 2), 1: ILLEGAL PARKING, 2: ATYPICAL
# ELEVATION (type = 3), 1: USER DEFINED, 2: ATYPICAL
# THERMAL (type = 4), 1: USER DEFINED, 2: ATYPICAL
# SOUND (type = 5), 1: HUMAN SCREAM, 2: EXPLOSION, 3: UNKNOWN
int8 abnormal_detail

# Position
float64 pos_x
float64 pos_y

