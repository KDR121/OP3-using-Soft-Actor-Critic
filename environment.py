#!/usr/bin/env python3
# coding: utf-8
import gym
from gym.core import ObservationWrapper
import numpy as np
import cv2
from pyquaternion import Quaternion
from torch.distributions.utils import lazy_property

import rospy
from rospy.topics import Publisher
from std_msgs import msg
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelState, ModelStates
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteModel, SpawnModel, SpawnModelRequest
from geometry_msgs.msg import Pose, Twist
import os

import pyautogui
import time
from icecream import ic

class Op3_Walking(gym.Env):
    """
    Op3を歩かせるタスクをする環境
    gym.Envに基づいて作ったつもりだけどちょっと違う。
    """
    def __init__(self):
        self.reward_class = Op3_Walking_reward()
        self.pubsub_class = Env_pubsub()
        self.gazebo_reset_class = Auto_gazebo_reset()
        self.action_num = 20 #jointの数
        self._max_episode_steps = 200
        self.sim_count = 0 #simulationの数をカウントする変数　可変
        self.max_sim_count = 10 #simkulationの最大回数
        #observation (min, max, (dimensions, ), shape)
        self.observation_space = gym.spaces.Box(-10, 10, (7,), np.float64)
        #action (最小値 , 最大値,　次元数, 型)
        self.action_space = gym.spaces.Box(-1, 1, (6,), np.float64)
        #----------------------------------------------------------------------------
        #環境ごとに変える
        self.init_observation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0])
        self.init_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.reset_position = np.array([0.0, 2.0, 0.0])
        self.state_observation = self.init_observation#初期値のサーボモータの角度
        self.state_action = self.init_action#行動の角度
        self.reset()
        return

    def reset(self): #Init pose
        """
        stateとrewardをreturnする
        各関節角の角度を0度にしてからgazeboのworldをリセットする。
        """
        #gazeboから各関節かくの角度を0にする
        rospy.sleep(0.5)
        self.pubsub_class.gazebo_reset()
        limit_move = 0.2
        #現在のactionの値と比較して一番差が大きいものを取得
        stock = self.state_action- self.init_action
        temp = max(stock)//limit_move
        loop_count = int(temp)
        if loop_count != 0:
            for i in range(loop_count):
                temp_action = self.state_action - ((self.state_action/(loop_count+1))*i)
                self.pubsub_class.publish_data_angle(temp_action)
        #差が大きいものとlimitの割り算の数だけループ
        #商をactionにして分割して送信
        self.pubsub_class.publish_data_angle(self.init_action)
        rospy.sleep(1.0)
        self.pubsub_class.gazebo_reset()
        rospy.sleep(1.0)
        self.state_action = self.init_action
        self.state_observation = self.init_observation
        observation = self.state_observation
        state_pose = self.pubsub_class.get_op3_pose()
        CheckDone = self.Check_continue(state_pose)
        if CheckDone == True:
            mode = 1
            if mode == 1:
                ic("動作を停止しました gazeboを再起動します")
                self.gazebo_reset_class.main()
            elif mode == 2:
                ic("動作が停止しました　再起動するには何かキー入力が必要です")
                input()
        return observation

    def step(self, action):
        """
        Input : action = np.array([20])
        Output : next_states: 動かした後の各関節角の値
                reward: 進んだ距離
                done: episodeを実行するかどうか、通常指定ステップ数を超えたらTrue,倒れたらTrueにもできる
        受け取るactionは動かす値(+0.1rad)なので現在のaction(例:2.3 + 0.1)に足し算して渡す
        """
        #動かす座標を取得して、現在の座標に足した値を返す
        action += self.state_action
        self.pubsub_class.publish_data_angle(action = action)
        self.state_action = self.pubsub_class.limit_action_space(self.state_action)
        state_pose = self.pubsub_class.get_op3_pose()
        # state_joints_angle = self.pubsub_class.get_op3_joints_angle()
        state_pose_vector = self.quaternion_to_vector(np.array([state_pose.orientation.x, state_pose.orientation.y, state_pose.orientation.z, state_pose.orientation.w]))
        self.state_observation = np.append(state_pose_vector, [self.state_action[0],self.state_action[1],self.state_action[2],self.state_action[3]])
        next_states = self.state_observation
        reward = self.reward_class.CalcReward(state_pose) #rewardを出力
        ic(reward)

        self.state_action = action #state_positionの更新
        if self.sim_count > self.max_sim_count:
            done = True
            self.sim_count = 0
        else:
            done = False
            self.sim_count += 1

        return next_states, reward, done

    def Check_continue(self, pose):
        if pose.position.x == 0 and pose.position.y == 0 and pose.position.z == 0:
            return True
        else:
            return False

    def max_episode_steps(self, steps):
        self._max_episode_steps = steps

    def set_max_sim_count(self, counts):
        self.max_sim_count = counts

    def quaternion_to_vector(self, q):
        q2 = Quaternion(q)
        return q2.vector

    def close(self):
        #環境を閉じる
        return


class Auto_gazebo_reset():
    """
    gazeboがcrashするのでマウス及びキーボードを操作し自動で再起動させるクラス
    """
    def main(self):
        pyautogui.moveTo(2851, 363, duration=2)
        pyautogui.click()
        pyautogui.hotkey('ctrl','c') #同時押し
        time.sleep(25)
        pyautogui.press('up')
        pos_x,pos_y = 860,1403
        pyautogui.moveTo(pos_x, pos_y, duration=2)
        pyautogui.press('enter')
        time.sleep(5)
        #画像検索して再生ボタンを押す
        pos_x,pos_y = 860,1403
        pyautogui.moveTo(pos_x, pos_y, duration=2)
        pyautogui.click()
        print(u"restart!")
        return

class Op3_Walking_reward():
    def __init__(self):
        self.pubsub_class = Env_pubsub()
        pass

    def set_target(self, target):#目標座標を決定
        self.target = target

    def CalcReward(self, pose):#報酬を設定
        """
        報酬を決める
        """
        if pose.position.x < 0:
            x = -np.sqrt(pose.position.x**2 + pose.position.y**2)/10
        else:
            x = np.sqrt(pose.position.x**2 + pose.position.y**2)/10
        return x


class Env_pubsub():
    """
    　publishとsubscribe関連を処理するクラス
    　subscribe : joint[20], position[4]
    　publish   : 20個のangle
    """
    def __init__(self):
        self.sac_pub = Float32MultiArray()
        self.pub00 = rospy.Publisher('/robotis_op3/head_pan_position/command', Float64, queue_size = 5)
        self.pub01 = rospy.Publisher('/robotis_op3/head_tilt_position/command', Float64, queue_size = 5)
        self.pub02 = rospy.Publisher('/robotis_op3/l_ank_pitch_position/command', Float64, queue_size = 5)
        self.pub03 = rospy.Publisher('/robotis_op3/l_ank_roll_position/command', Float64, queue_size = 5)
        self.pub04 = rospy.Publisher('/robotis_op3/l_el_position/command', Float64, queue_size = 5)
        self.pub05 = rospy.Publisher('/robotis_op3/l_hip_pitch_position/command', Float64, queue_size = 5)
        self.pub06 = rospy.Publisher('/robotis_op3/l_hip_roll_position/command', Float64, queue_size = 5)
        self.pub07 = rospy.Publisher('/robotis_op3/l_hip_yaw_position/command', Float64, queue_size = 5)
        self.pub08 = rospy.Publisher('/robotis_op3/l_knee_position/command', Float64, queue_size = 5)
        self.pub09 = rospy.Publisher('/robotis_op3/l_sho_pitch_position/command', Float64, queue_size = 5)
        self.pub10 = rospy.Publisher('/robotis_op3/l_sho_roll_position/command', Float64, queue_size = 5)
        self.pub11 = rospy.Publisher('/robotis_op3/r_ank_pitch_position/command', Float64, queue_size = 5)
        self.pub12 = rospy.Publisher('/robotis_op3/r_ank_roll_position/command', Float64, queue_size = 5)
        self.pub13 = rospy.Publisher('/robotis_op3/r_el_position/command', Float64, queue_size = 5)
        self.pub14 = rospy.Publisher('/robotis_op3/r_hip_pitch_position/command', Float64, queue_size = 5)
        self.pub15 = rospy.Publisher('/robotis_op3/r_hip_roll_position/command', Float64, queue_size = 5)
        self.pub16 = rospy.Publisher('/robotis_op3/r_hip_yaw_position/command', Float64, queue_size = 5)
        self.pub17 = rospy.Publisher('/robotis_op3/r_knee_position/command', Float64, queue_size = 5)
        self.pub18 = rospy.Publisher('/robotis_op3/r_sho_pitch_position/command', Float64, queue_size = 5)
        self.pub19 = rospy.Publisher('/robotis_op3/r_sho_roll_position/command', Float64, queue_size = 5)
        self.p = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size= 5)
        rospy.Subscriber('/robotis_op3/joint_states', JointState , self.callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback2)

    def callback(self, array):
        self.joints_angle = array.position

    def callback2(self, array):
        self.op3_pose = array.pose[1]

    def get_op3_pose(self):
        return self.op3_pose

    def get_op3_joints_angle(self):
        return self.joints_angle


    def gazebo_reset(self):
        """
            gazeboをresetする関数
            1. reset gazebo & model
            2. delete model & spawn model
            3. execution roslaunch
            1のみ問題なく動く
        """
        reset_pattern = 1
        if reset_pattern == 1:
            #------------------------------------------------
            # reset gazebo
            rospy.wait_for_service("/gazebo/reset_world")
            rospy.ServiceProxy("/gazebo/reset_world", Empty).call()
            rospy.wait_for_service("/gazebo/delete_model")
        elif reset_pattern == 2:
            #------------------------------------------------
            # op3を消して召喚する
            # 問題:リセットした後にモデルがコントロールできない
            srv = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            model_name = "robotis_op3"
            srv(model_name)
            rospy.wait_for_service("/gazebo/spawn_urdf_model")
            spawn_urdf = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
            pose = Pose()
            pose.position.x = 0
            pose.position.y = 0
            pose.position.z = 0.285
            description_xml = ''
            description_xml = '~/catkin_ws/src/pro_op3_config/script/op3_urdf/robotis_op3.urdf.xacro'
            p = os.popen("rosrun xacro xacro " + description_xml)
            xml_string = p.read()
            p.close()
            # spawn new model
            req = SpawnModelRequest()
            req.model_name = 'robotis_op3' # model name from command line input
            req.model_xml = xml_string
            req.initial_pose = pose
            spawn_urdf(req)
            os.popen("roslaunch op3_gazebo position_controller.launch")
        elif reset_pattern == 3:
            #---------------------------------------------------
            # roslaunchをする
            # 問題：うまく前のroslaunchをkillできない
            os.popen("roslaunch op3_gazebo robotis_world.launch")

    def publish_data_angle(self, action):
        """
         SACのpublishする関数
         20個の関節角を-3.14~3.14の範囲で動かす。
        """
        publish_limit = 0.2 #一度に動かしてもよい限界値
        action = self.limit_action_space(action)
        pub = Float64()
        #固定値
        pub.data = 1.5
        self.pub04.publish(pub)
        pub.data = -1.5
        self.pub13.publish(pub)
        pub.data = -0.1
        self.pub08.publish(pub)
        self.pub07.publish(pub)
        pub.data = 0.1
        self.pub15.publish(pub)
        self.pub16.publish(pub)
        pub.data = -0.12
        self.pub03.publish(pub)
        pub.data = 0.12
        #可変値
        self.pub12.publish(pub)
        pub.data = action[0]
        self.pub05.publish(pub)
        pub.data = action[1]
        self.pub08.publish(pub)
        pub.data = action[2]
        self.pub14.publish(pub)
        pub.data = action[3]
        self.pub17.publish(pub)
        pub.data = action[4]
        self.pub09.publish(pub)
        pub.data = action[5]
        self.pub18.publish(pub)
        ic(action)
        rospy.sleep(0.025)

    def publish_data_pose(self, pose):
        pub = ModelState()
        pub.model_name = 'robotis_op3'
        pub.pose.position.x = pose[0]
        pub.pose.position.y = pose[1]
        pub.pose.position.z = pose[2]
        self.p.publish(pub)

    def limit_action_space(self, action):
        """
        送信されてきたactionに制限をかける
        """
        limit = 0.9
        for i, a in enumerate(action):
            if a > limit:
                action[i] = limit
            if a < -(limit):
                action[i] = -(limit)
        return action
