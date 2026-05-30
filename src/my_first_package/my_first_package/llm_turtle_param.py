#!/usr/bin/env python3
import os
import sys
import json
import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from std_srvs.srv import Empty  # for turtle_sim background updates
from openai import OpenAI


class LlmTurtleParam(Node):
    def __init__(self):
        super().__init__('llm_turtle_param')
        
        # 1. Create a service client to control turtlesim parameters remotely.
        self.param_client = self.create_client(SetParameters, '/turtlesim/set_parameters')
        
        # 2. Create a service client to refresh the background color after updates.
        self.clear_client = self.create_client(Empty, '/clear')
        
        # Wait until the service server is available.
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the /turtlesim parameter service...')
            
        self.client = OpenAI()
        self.get_logger().info('=== LLM turtlesim parameter agent started ===')
        self.get_logger().info('Example commands: "Make the background warm pink", "Change it to chalkboard green"')

    def ask_llm_and_set_param(self, user_command):
        """
        Convert natural language into RGB values and update turtlesim parameters.
        """
        self.get_logger().info(f"Received environment control command: '{user_command}'")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "set_background_color",
                    "description": "Change the RGB background color parameters of the turtlesim simulator.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "r": {"type": "integer", "description": "Red value (0 to 255)"},
                            "g": {"type": "integer", "description": "Green value (0 to 255)"},
                            "b": {"type": "integer", "description": "Blue value (0 to 255)"},
                            "reasoning": {"type": "string", "description": "English explanation for the selected color combination"}
                        },
                        "required": ["r", "g", "b", "reasoning"]
                    }
                }
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a spatial environment control system that maps human emotions or descriptions to numeric RGB color values."},
                    {"role": "user", "content": user_command}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "set_background_color"}}
            )

            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            
            r_val = int(arguments.get("r", 69))
            g_val = int(arguments.get("g", 86))
            b_val = int(arguments.get("b", 255))
            reason = arguments.get("reasoning", "No explanation provided")

            self.get_logger().info(f"AI guidance: {reason}")
            self.get_logger().info(f"Selected parameter values -> R: {r_val}, G: {g_val}, B: {b_val}")

            # 3. Build the ROS 2 parameter update request.
            req = SetParameters.Request()
            
            # Create and add each RGB parameter object.
            p_r = Parameter(name='background_r', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=r_val))
            p_g = Parameter(name='background_g', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=g_val))
            p_b = Parameter(name='background_b', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=b_val))
            
            req.parameters = [p_r, p_g, p_b]

            # Call the parameter service and wait for completion.
            future = self.param_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            
            # 4. Refresh the background after applying the parameters.
            clear_req = Empty.Request()
            clear_future = self.clear_client.call_async(clear_req)
            rclpy.spin_until_future_complete(self, clear_future)
            
            self.get_logger().info("Turtlesim parameters updated successfully.")

        except Exception as e:
            self.get_logger().error(f"An error occurred: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = LlmTurtleParam()

    try:
        while rclpy.ok():
            user_input = input("\n[Describe the desired background mood (q to quit)]: ")
            if user_input.lower() == 'q':
                break
            if user_input.strip() == '':
                continue
                
            node.ask_llm_and_set_param(user_input)
            
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
