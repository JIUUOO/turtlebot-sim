#!/usr/bin/env python3
import os
import sys
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from openai import OpenAI


class LlmTurtleMotion(Node):
    def __init__(self):
        # Initialize the ROS 2 node.
        super().__init__('llm_turtle_motion')
        
        # Create a publisher for the turtlesim velocity command topic.
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
 
        self.client = OpenAI()
        
        self.get_logger().info('=== LLM turtlesim motion agent node started ===')
        self.get_logger().info('Example commands: "Move forward 2 meters", "Rotate clockwise 90 degrees"')

    def ask_llm_and_move(self, user_command):
        """
        Analyze a natural-language command with the OpenAI API, then drive turtlesim.
        """
        self.get_logger().info(f"Analyzing user command: '{user_command}'")

        # Define the function schema that the LLM should fill.
        # This makes the LLM return structured JSON instead of free-form text.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_robot",
                    "description": "Control the robot's linear velocity and angular velocity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "linear_velocity": {
                                "type": "number",
                                "description": "Positive moves forward in m/s, negative moves backward, 0 stops."
                            },
                            "angular_velocity": {
                                "type": "number",
                                "description": "Positive turns counterclockwise in rad/s, negative turns clockwise, 0 stops."
                            }
                        },
                        "required": ["linear_velocity", "angular_velocity"]
                    }
                }
            }
        ]

        try:
            # Call the OpenAI Chat Completions API.
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI robot interface that converts human commands into robot control parameters."},
                    {"role": "user", "content": user_command}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "control_robot"}} # Force this function call.
            )

            # Extract the function arguments filled by the LLM.
            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            
            linear_v = arguments.get("linear_velocity", 0.0)
            angular_v = arguments.get("angular_velocity", 0.0)

            self.get_logger().info(f"LLM result -> linear velocity: {linear_v} m/s, angular velocity: {angular_v} rad/s")

            # Create and publish the ROS 2 message.
            twist_msg = Twist()
            twist_msg.linear.x = float(linear_v)   # Forward/backward velocity.
            twist_msg.angular.z = float(angular_v) # Rotation velocity.
            
            # Publish the command to turtlesim.
            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info("ROS 2 topic published successfully.")

        except Exception as e:
            self.get_logger().error(f"An error occurred: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = LlmTurtleMotion()

    try:
        while rclpy.ok():
            user_input = input("\n[Enter a command (q to quit)]: ")
            if user_input.lower() == 'q':
                break
            if user_input.strip() == '':
                continue
                
            # Interpret the natural-language command and move the robot.
            node.ask_llm_and_move(user_input)
            
            # Process ROS 2 events briefly.
            rclpy.spin_once(node, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
