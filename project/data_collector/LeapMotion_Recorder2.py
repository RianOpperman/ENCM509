################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################
import pandas as pd
import numpy as np

'''
import sys
sys.path.append("../lib/Leap")
sys.path.append("../lib/")
sys.path.append("../lib/x86")
'''


import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    def on_init(self, controller):
        print("Initialized")

        # Create DataFrame
        columns = ['Frame id', 'timestamp', 'hands', 'fingers', 'tools', 'gestures',
                   'handType', 'handID', 'handPalmPosition', 'pitch', 'roll', 'yaw',
                   'armDirection', 'wristPosition', 'elbowPosition',
                   'ToolID', 'ToolTipPosition', 'ToolDirection',
                   'GestureID', 'GestureType', 'GestureCircleClockwiseness', 'GestureCircleSweptAngle',
                   'GestureCircleProgress', 'GestureCircleRadius', 'GestureStateNames',
                   'SwipePosition', 'SwipeDirection', 'SwipeSpeed',
                   'KeytapPosition', 'KeytapDirection',
                   'ScreentapPosition', 'ScreentapDirection']
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

        for finger in finger_names:
            columns.append(finger + 'ID')
            columns.append(finger + 'Length')
            columns.append(finger + 'Width')

        for finger in finger_names:
            for bone in bone_names:
                columns.append(finger + bone + 'Start')
                columns.append(finger + bone + 'End')
                columns.append(finger + bone + 'Direction')

        self.df = pd.DataFrame(index=np.empty(0), columns=columns)
        self.df = self.df.fillna(0)
        # index for DataFrame
        self.loc_i = 0

    def on_connect(self, controller):
        print("Connected")

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE);
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP);
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE);

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        # print(self.df)

        # compression_opts = dict(method='zip',archive_name='out.csv')
        temp = int(time.time())
        name = "MA_gibb_%d.csv" % (temp)
        self.df.to_csv(name, index=False)  # ,compression=compression_opts)

        print("Data was saved")
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        self.df.loc[self.loc_i] = [frame.id, frame.timestamp, len(frame.hands), len(frame.fingers),
                                   len(frame.tools), len(frame.gestures()),
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0,
                                   0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0]

        print("Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
            frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures())))

        # Get hands
        for hand in frame.hands:

            # 'handType'
            handType = "Left hand" if hand.is_left else "Right hand"
            self.df.iloc[self.loc_i, self.df.columns.get_loc('handType')] = handType

            # 'handID', 'handPalmPosition'
            print("  %s, id %d, position: %s" % (
                handType, hand.id, hand.palm_position))
            self.df.iloc[self.loc_i, self.df.columns.get_loc('handID')] = hand.id
            self.df.iloc[self.loc_i, self.df.columns.get_loc('handPalmPosition')] = hand.palm_position

            # 'pitch', 'roll', 'yaw'
            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction
            # Calculate the hand's pitch, roll, and yaw angles
            print("  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
                direction.pitch * Leap.RAD_TO_DEG,
                normal.roll * Leap.RAD_TO_DEG,
                direction.yaw * Leap.RAD_TO_DEG))
            self.df.iloc[self.loc_i, self.df.columns.get_loc('pitch')] = direction.pitch * Leap.RAD_TO_DEG
            self.df.iloc[self.loc_i, self.df.columns.get_loc('roll')] = normal.roll * Leap.RAD_TO_DEG
            self.df.iloc[self.loc_i, self.df.columns.get_loc('yaw')] = direction.yaw * Leap.RAD_TO_DEG

            # 'armDirection', 'wristPosition', 'elbowPosition'
            # Get arm bone
            arm = hand.arm
            print("  Arm direction: %s, wrist position: %s, elbow position: %s" % (
                arm.direction,
                arm.wrist_position,
                arm.elbow_position))
            self.df.iloc[self.loc_i, self.df.columns.get_loc('armDirection')] = arm.direction,
            self.df.iloc[self.loc_i, self.df.columns.get_loc('wristPosition')] = arm.wrist_position
            self.df.iloc[self.loc_i, self.df.columns.get_loc('elbowPosition')] = arm.elbow_position

            # Get fingers
            for finger in hand.fingers:
                print("   %s finger, id: %d, length: %fmm, width: %fmm" % (
                    self.finger_names[finger.type],
                    finger.id,
                    finger.length,
                    finger.width))
                # 'ID', 'Length', 'Width' for each finger in hand.fingers
                self.df.iloc[self.loc_i, self.df.columns.get_loc(self.finger_names[finger.type] + 'ID')] = finger.id
                self.df.iloc[
                    self.loc_i, self.df.columns.get_loc(self.finger_names[finger.type] + 'Length')] = finger.length
                self.df.iloc[
                    self.loc_i, self.df.columns.get_loc(self.finger_names[finger.type] + 'Width')] = finger.width

                # 'start', 'end', 'direction' for each bone for each finger
                # Get bones
                for b in range(0, 4):
                    bone = finger.bone(b)
                    print("      Bone: %s, start: %s, end: %s, direction: %s" % (
                        self.bone_names[bone.type],
                        bone.prev_joint,
                        bone.next_joint,
                        bone.direction))

                    self.df.iloc[self.loc_i, self.df.columns.
                        get_loc(self.finger_names[finger.type] +
                                self.bone_names[bone.type] + 'Start')] = bone.prev_joint
                    self.df.iloc[self.loc_i, self.df.columns.
                        get_loc(self.finger_names[finger.type] +
                                self.bone_names[bone.type] + 'End')] = bone.next_joint
                    self.df.iloc[self.loc_i, self.df.columns.
                        get_loc(self.finger_names[finger.type] +
                                self.bone_names[bone.type] + 'Direction')] = bone.direction

                    # 'ToolID','ToolTipPosition','ToolDirection'
        # Get tools
        for tool in frame.tools:
            print("  Tool id: %d, position: %s, direction: %s" % (
                tool.id, tool.tip_position, tool.direction))
            self.df.iloc[self.loc_i, self.df.columns.get_loc('ToolID')] = tool.id
            self.df.iloc[self.loc_i, self.df.columns.get_loc('ToolTipPosition')] = tool.tip_position
            self.df.iloc[self.loc_i, self.df.columns.get_loc('ToolDirection')] = tool.direction

        # Get gestures
        for gesture in frame.gestures():
            # 'GestureID','GestureType', 'GestureStateNames'
            self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureType')] = gesture.type
            self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureID')] = gesture.id
            self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureStateNames')] = self.state_names[gesture.state]

            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                circle = CircleGesture(gesture)
                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI / 2:
                    clockwiseness = "clockwise"
                else:
                    clockwiseness = "counterclockwise"

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state != Leap.Gesture.STATE_START:
                    previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
                    swept_angle = (circle.progress - previous_update.progress) * 2 * Leap.PI

                # 'GestureCircleClockwiseness', 'GestureCircleSweptAngle','GestureCircleProgress','GestureCircleRadius'
                print("  Circle id: %d, %s, progress: %f, radius: %f, angle: %f degrees, %s" % (
                    gesture.id, self.state_names[gesture.state],
                    circle.progress, circle.radius, swept_angle * Leap.RAD_TO_DEG, clockwiseness))
                self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureCircleProgress')] = circle.progress
                self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureCircleRadius')] = circle.radius
                self.df.iloc[
                    self.loc_i, self.df.columns.get_loc('GestureCircleSweptAngle')] = swept_angle * Leap.RAD_TO_DEG
                self.df.iloc[self.loc_i, self.df.columns.get_loc('GestureCircleClockwiseness')] = clockwiseness

            # 'SwipePosition','SwipeDirection','SwipeSpeed'
            if gesture.type == Leap.Gesture.TYPE_SWIPE:
                swipe = SwipeGesture(gesture)
                print("  Swipe id: %d, state: %s, position: %s, direction: %s, speed: %f" % (
                    gesture.id, self.state_names[gesture.state],
                    swipe.position, swipe.direction, swipe.speed))
                self.df.iloc[self.loc_i, self.df.columns.get_loc('SwipePosition')] = swipe.position
                self.df.iloc[self.loc_i, self.df.columns.get_loc('SwipeDirection')] = swipe.direction
                self.df.iloc[self.loc_i, self.df.columns.get_loc('SwipeSpeed')] = swipe.speed

            # 'KeytapPosition','KeytapDirection'
            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                keytap = KeyTapGesture(gesture)
                print("  Key Tap id: %d, %s, position: %s, direction: %s" % (
                    gesture.id, self.state_names[gesture.state],
                    keytap.position, keytap.direction))
                self.df.iloc[self.loc_i, self.df.columns.get_loc('KeytapPosition')] = keytap.position
                self.df.iloc[self.loc_i, self.df.columns.get_loc('KeytapDirection')] = keytap.direction

            # 'ScreentapPosition','ScreentapDirection'
            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
                screentap = ScreenTapGesture(gesture)
                print("  Screen Tap id: %d, %s, position: %s, direction: %s" % (
                    gesture.id, self.state_names[gesture.state],
                    screentap.position, screentap.direction))
                self.df.iloc[self.loc_i, self.df.columns.get_loc('ScreentapPosition')] = screentap.position
                self.df.iloc[self.loc_i, self.df.columns.get_loc('ScreentapDirection')] = screentap.direction

        if not (frame.hands.is_empty and frame.gestures().is_empty):
            print("")

        self.loc_i += 1

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline() # uncomment for console applications
        #raw_input() # uncomment for Jupyter Notebook
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
if __name__ == "__main__":
    main()