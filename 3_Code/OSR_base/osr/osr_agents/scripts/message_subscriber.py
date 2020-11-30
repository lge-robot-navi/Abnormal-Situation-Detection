

import rospy
import message_filters

class MessageAsyncSubscriber:
    def __init__(self, sub_pose_topic):
        self.sub_pose = rospy.Subscriber(sub_pose_topic, Odometry, self.pose_callback)
        self.pose_msg = None

    def pose_callback(self, msg):
        self.pose_msg = copy.deepcopy(msg)

    def get_messages(self):
        return self.pose_msg

class MessageSyncSubscriber:
    def __init__(self):
        sub_point_cloud_topic = rospy.get_param('~sub_point_cloud_topic', "/velodyne_points")
        sub_pose_topic = rospy.get_param('~sub_pose_topic', "/pose_imu")
        self.sub_pose = message_filters.Subscriber(sub_pose_topic, Odometry)
        self.sub_point_cloud = message_filters.Subscriber(sub_point_cloud_topic, PointCloud2)
        self.pose_msg = None
        self.point_cloud_msg = None

        self.ts = message_filters.TimeSynchronizer([self.sub_pose, self.sub_point_cloud], 10)
        self.ts.registerCallback(self.callback)
        self.is_messages = False

    def callback(self, pose_msg, point_cloud_msg):
        self.pose_msg = copy.deepcopy(pose_msg)
        self.point_cloud_msg = copy.deepcopy(point_cloud_msg)
        self.messages_ok = True

    def get_messages(self):
        if (self.pose_msg is not None) and (self.point_cloud_msg is not None):
            return self.pose_msg, self.point_cloud_msg
        return None, None

def main():
    rospy.init_node("sound_abnormal_detection")
    try:
        period = rospy.get_param('~period', 10)
        robot_id = rospy.get_param('~robot_id', 1)
        pub_abnormal_topic = rospy.get_param('~pub_abnormal_topic', '/osr/sound_abnormal')
        sub_pose_topic = rospy.get_param('~sub_pose_topic', "/pose_imu")
    except KeyError:
        rospy.logerr("[ERROR] ROS parameters cannot be loaded.")

    # Publisher Definition
    pub_abnormal = rospy.Publisher(pub_abnormal_topic, Abnormal, queue_size=10)
    # Subscriber Definition
    message_subscriber = MessageSubscriber(sub_pose_topic)

    rate = rospy.Rate(period)
    while not rospy.is_shutdown():
        pose_msg = message_subscriber.get_messages()
        if pose_msg is None:
            rospy.logerr("[ERROR] Data is not subscribed yet.")
            continue

        # Code Here #################################

        #############################################

        abnormal = Abnormal()
        abnormal.agent_id = robot_id
        abnormal.report_id = rospy.get_time()
        abnormal.pos_x = pose_msg.pose.pose.position.x
        abnormal.pos_y = pose_msg.pose.pose.position.y

        # input the result from this module
        abnormal.status = 0
        abnormal.type = 5
        abnormal.detail = 0
        abnormal.score = 0
        pub_abnormal.publish(abnormal)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
