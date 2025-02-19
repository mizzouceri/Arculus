# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:09:05 2024

@author: bvlsc
"""

import imageio
import numpy as np
import random
import pygame
import sys
import time
import math


start=time.time()


rewards_ratio = [3,4,5,6]
# rewards_ratio = [10]
trust_zone_radii = [0, 250]
fullBattery = 5000000



for tzr in trust_zone_radii:
    for rr in rewards_ratio:
    
    
        # Initialize Pygame
        pygame.init()
        
        # Set the dimensions of the window
        width, height = 800, 800
        window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Truck and Points Visualization with Q-learning")
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        CYAN = (0, 255, 255)
        ORANGE = (255, 165, 0)
        YELLOW = (255, 255, 0)
        
        cheese_eaten = False
        vx, vy = 0, 0
        
        # Define obstacles as rectangles
        obstacles = [
            pygame.Rect(370, 300, 400, 100),
            pygame.Rect(175, 200, 100, 200),
            pygame.Rect(500 - 50, 700 - 50, 100, 100)
        ]
        
        congestion = [
            pygame.Rect(75, 75, 100, 100)
        ]
        
        
        window_size = 10
        
        def save_plot_reward_to_file(plot_reward, file_name="plot_reward.txt"):
            with open(file_name, 'w') as file:
                for reward in plot_reward:
                    file.write(f"{reward}\n")
            # print(f"Plot rewards saved to {file_name}")
        
        
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        
        # Generate random points within the window bounds
        def generate_random_point():
            random.seed(69)
            return (random.randint(100, width-100), random.randint(100, height-100))
            # return (width // 2, height // 2)
        
        # Q-learning parameters
        alpha = 0.2
        gamma = 0.95
        # epsilon = 0.2
        
        
        episodes = 10000
        testing = 10000
        
        
        

        
        
        
        # Initialize video recording for first two and last two episodes
        capture_episodes = []
        video_writers = {}
        
        def start_recording(episode):
            video_writers[episode] = imageio.get_writer(f'episode_{episode}.mp4', fps=30)
        
        def record_frame(episode):
            pixels = pygame.surfarray.array3d(window)
            pixels = np.transpose(pixels, (1, 0, 2))  # Adjust for correct orientation
            video_writers[episode].append_data(pixels)
        
        def stop_recording(episode):
            if episode in video_writers:
                video_writers[episode].close()
        
        
        
        
        plot_reward = []
        
        # Initialize Q-table
        states = []
        
        for x in range(0, width, 20):
            for y in range(0, height, 20):
                for cheese_eaten in [True, False]:
                    for near_congestion in [True, False]:
                        for s in ['left', 'right', 'up', 'down', 'down_left', 'down_right', 'up_left', 'up_right', None]:
                            for near_customer1 in [True, False]:
                                for near_customer2 in [True, False]:
                                    for near_depot in [True, False]:
                                        for near_emergency in [True, False]:
                                            for batteryLow1 in [True, False]:
                                                for batteryLow2 in [True, False]:
                                                    states.append((
                                                        ((x, y), (x, y)), # Drones' positions
                                                        cheese_eaten,
                                                        (batteryLow1, batteryLow2),
                                                        (near_congestion, s),
                                                        (near_customer1, near_customer2, near_depot, near_emergency)
                                                    ))
                                                
        # print(states)
        actions = [
            'move_to_customer', 'move_to_depot', 'move_to_emergency',
            'move_up', 'move_down', 'move_left', 'move_right',
            'move_up_left', 'move_up_right', 'move_down_left', 'move_down_right',
            'increase_speed', 'decrease_speed', 
            'move_away'
        ]
        q_table = {}
        for state in states:
            q_table[state] = {action: 0 for action in actions}
        
        def is_in_congestion_zone(drone):
            for cong in congestion:
                if cong.collidepoint(drone):
                    return True
            return False
        
        def is_near_congestion(drone):
            if not congestion:
                return False, None
        
            nearest_congestion = min(congestion, key=lambda cong: distance(drone, (cong.x + cong.width // 2, cong.y + cong.height // 2)))
            nearest_distance = distance(drone, (nearest_congestion.x + nearest_congestion.width // 2, nearest_congestion.y + nearest_congestion.height // 2))
            
            # Round to the nearest multiple of 20
            nearest_distance = round(nearest_distance / 20) * 20
            near = nearest_distance < 100  # Consider near if within 100 units
        
            # Calculate the center of the nearest congestion zone
            congestion_center_x = nearest_congestion.x + nearest_congestion.width // 2
            congestion_center_y = nearest_congestion.y + nearest_congestion.height // 2
        
            # Determine the primary direction of the nearest congestion zone
            if congestion_center_x < drone[0] and congestion_center_y < drone[1]:
                direction = 'up_left'
            elif congestion_center_x > drone[0] and congestion_center_y < drone[1]:
                direction = 'up_right'
            elif congestion_center_x < drone[0] and congestion_center_y > drone[1]:
                direction = 'down_left'
            elif congestion_center_x > drone[0] and congestion_center_y > drone[1]:
                direction = 'down_right'
            elif congestion_center_x < drone[0]:
                direction = 'left'
            elif congestion_center_x > drone[0]:
                direction = 'right'
            elif congestion_center_y < drone[1]:
                direction = 'up'
            else:
                direction = 'down'
        
            if near:
                return near, direction
            return near, None
        
        # Function to move the drone towards a target
        def move_towards(drone, target, speed=2):
            global vx, vy
            if target is None:
                vx, vy = 0, 0
                return drone
        
            x1, y1 = drone
            x2, y2 = target
        
            dx, dy = x2 - x1, y2 - y1
            distance = np.hypot(dx, dy)
        
            if distance < speed:
                vx, vy = 0, 0
                return target
        
            vx, vy = (dx / distance) * speed, (dy / distance) * speed
            x1 += vx
            y1 += vy
        
            return x1, y1
        
        def distance(point1, point2):
            return np.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)
        
        # Function to check if within radius
        def is_within_radius(point1, point2, radius=20):
            return np.hypot(point1[0] - point2[0], point1[1] - point2[1]) <= radius
        
        # Function to get reward
        def get_reward(drone1, customer1, emergency, depot, delivered1, truck, speed1, near_customer1, near_depot, near_emergency, battery1):
            reward = 0
        
            for obstacle in obstacles:
                if obstacle.collidepoint(drone1):
                    return -100
            
            if battery1 <= 0:
                return -10000
        
            # Reward for reaching the respective customer
            if not delivered1 and is_within_radius(drone1, customer1):
                reward += 50 * rr

        
            # Reward for reaching depot or emergency after delivery
            if is_within_radius(drone1, depot) or is_within_radius(drone1, emergency):
                reward += 50
        
            # Penalty for being in congestion zones
            if is_in_congestion_zone(drone1):
                reward -= 50
        
            # Velocity penalty for both drones
            reward -= 0.0005 * (abs(vx) + abs(vy))  
        
            return reward
        
        def min_max_scaling(q_values):
            min_q = min(q_values)
            max_q = max(q_values)
            if max_q == min_q:
                return [0.5 for _ in q_values]  # Return 0.5 for all elements if all Q-values are the same
            return [(q - min_q) / (max_q - min_q) for q in q_values]
        
        
        def softmax(q_values):
            # print("Q", q_values)
            max_q = max(q_values)
            temperature = 0.7
            exp_values = [math.exp((q-max_q) / temperature) for q in q_values]
            # print("E", exp_values)
            total = sum(exp_values)
            probabilities = [exp / total for exp in exp_values]
            # print("P", probabilities)
            return probabilities
            
        # Function to choose an action
        def choose_action(state, epsilon):
            if random.uniform(0, 1) < epsilon:
                return random.choice(actions)
            else:
                q_values = [q_table[state][action] for action in actions]
                # print("Before scale", q_values)
                scaled = min_max_scaling(q_values)
                probabilities = softmax(scaled)
                # probabilities = softmax(q_values)
                return random.choices(actions, probabilities)[0]
                # return max(q_table[state], key=q_table[state].get)
        
        def generate_dynamic_congestion(episode):
            congestion.clear()
            num_congestion_zones = random.randint(1, 4)  # Adjust the number of zones as needed
            for _ in range(num_congestion_zones):
                x = random.randint(0, (width // 20) - 5) * 20  # Generate x-coordinate aligned to multiple of 20
                y = random.randint(0, (height // 20) - 5) * 20
                congestion.append(pygame.Rect(x, y, 100, 100))
        
        def add_random_congestion(congestion_list, width, height):
            new_x = random.randint(0, (width // 20) - 5) * 20
            new_y = random.randint(0, (height // 20) - 5) * 20
            
            new_congestion = pygame.Rect(new_x, new_y, 100, 100)
            congestion_list.append(new_congestion)
        
        def move_away_from_congestion(state):
            drone = state[0]
            nearest_congestion = min(congestion, key=lambda cong: distance(drone, (cong.x + cong.width // 2, cong.y + cong.height // 2)))
            #nearest_distance = distance(drone, (nearest_congestion.x + nearest_congestion.width // 2, nearest_congestion.y + nearest_congestion.height // 2))
        
            congestion_center_x = nearest_congestion.x + nearest_congestion.width // 2
            congestion_center_y = nearest_congestion.y + nearest_congestion.height // 2
        
            # Determine the primary direction of the nearest congestion zone
            direction = (0, 0)
            if congestion_center_x < drone[0] and congestion_center_y < drone[1]:
                direction = (-1, -1)
            elif congestion_center_x > drone[0] and congestion_center_y < drone[1]:
                direction = (1, -1)
            elif congestion_center_x < drone[0] and congestion_center_y > drone[1]:
                direction = (-1, 1)
            elif congestion_center_x > drone[0] and congestion_center_y > drone[1]:
                direction = (1, 1)
            elif congestion_center_x < drone[0]:
                direction = (-1, 0)
            elif congestion_center_x > drone[0]:
                direction = (1, 0)
            elif congestion_center_y < drone[1]:
               direction = (0, -1)
            else:
                direction = (0, 1)
        
            # print(directionOfZone)
            # Calculate the new position
            new_x = drone[0] + direction[0] * -50
            new_y = drone[1] + direction[1] * -50
        
            # Ensure the new position is within bounds
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
        
            return (new_x, new_y), direction
        
                
        
        
        # Font for displaying text
        font = pygame.font.Font(None, 36)
        total_rewards = []
        delivered_rewards = []
        path = set()
        # Training the Q-learning agent with visualization
        
        time_congestion=0
        for episode in range(episodes):
            # print("Tr. Ep.", episode)
            time_congestion=0
            
            congestion.clear()
            generate_dynamic_congestion(episode)
            if episode % 20000 == 0:
                congestion.pop()
                new_congestion = pygame.Rect(280, 300, 100, 100)
                congestion.append(new_congestion)
            path = set()
            drone = (100, 100)
            truck = generate_random_point()
            customer = (100, 700)
            depot = (700, 700)
            emergency = (700, 150)
            
            drone2 = (200, 200)  # Initial position of the second drone
            customer2 = (300, 300)  # Position of the second customer
            
            
            # epsilon = 0.2
            epsilon = max(0.01, 1 - episode / (0.95*episodes)) 
            delivered = False
            near_customer = False
            near_depot = False
            near_emergency = False
            target = None
            steps = 0
            viz_training = True
            viz_step=99
            total_reward = 0
            speed = 1
            state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (distance(drone, customer) < 50, distance(drone, depot) < 50, distance(drone, emergency) < 50))
            dir = None
    
            battery = fullBattery
            while not is_within_radius(drone, depot) and not is_within_radius(drone, emergency) and battery > 0:
                if steps % 200 == 0:
                    if steps != 0:
                        random.shuffle(congestion)
                        congestion.pop()
                    add_random_congestion(congestion, 800, 800)
                # epsilon *= 0.9995
                if state not in q_table:
                    q_table[state] = {action: 0 for action in actions}
                action = choose_action(state, epsilon)
                dronePrev = drone
                
                if action == 'move_to_customer':
                    target = customer
                    drone = move_towards(drone, target, speed)
                elif action == 'move_to_depot':
                    target = depot
                    drone = move_towards(drone, target, speed)
                elif action == 'move_to_emergency':
                    target = emergency
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up':
                    target = (drone[0], max(drone[1] - 20, 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down':
                    target = (drone[0], min(drone[1] + 20, height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_left':
                    target = (max(drone[0] - 20, 0), drone[1])
                    drone = move_towards(drone, target, speed)
                elif action == 'move_right':
                    target = (min(drone[0] + 20, width-1), drone[1])
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up_left':
                    target = (max(drone[0] - math.sqrt(20), 0), max(drone[1] - math.sqrt(20), 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up_right':
                    target = (min(drone[0] + math.sqrt(20), width-1), max(drone[1] - math.sqrt(20), 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down_left':
                    target = (max(drone[0] - math.sqrt(20), 0), min(drone[1] + math.sqrt(20), height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down_right':
                    target = (min(drone[0] + math.sqrt(20), width-1), min(drone[1] + math.sqrt(20), height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'increase_speed':
                    if speed < 20:
                        speed += 1
                    else:
                        speed = 20
                elif action == 'decrease_speed':
                    if speed > 1:
                        speed -= 1
                    else:
                        speed = 1
                elif action == 'move_away':
        
                    target, dir = move_away_from_congestion(state)
                    drone = move_towards(drone, target, speed)
                
                

            
                battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
                isBatteryLow = False if battery / fullBattery > 0.25 else True
                # battery = round(battery / 5) * 5
                
                # print(battery)
            
                reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)
        
                if battery < 0:
                    battery = 0
        
                for obstacle in obstacles:
                    if obstacle.collidepoint(drone):
                        drone = dronePrev
                
                path.add(drone)
                # if action == 'move_to_customer':
                #     print(reward)
                
                if is_in_congestion_zone(drone):
                    time_congestion+=1
                
                if distance(drone, customer) < 50:
                    near_customer = True
            
                if distance(drone, depot) < 50:
                    near_depot = True
        
                if distance(drone, emergency) < 50:
                    near_emergency = True
                
                if not delivered and is_within_radius(drone, customer):
                    # print("delivered")
                    delivered = True
                
            
                total_reward += reward
                next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, isBatteryLow, is_near_congestion(drone), (near_customer, near_depot, near_emergency))
                if next_state not in q_table:
                    q_table[next_state] = {action: 0 for action in actions}
                # if cheese_eaten:
                #     print(next_state)
                q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
                state = next_state
    
                
                steps += 1
                
    
                
            
        # print("Training Done")
        
        testing_rewards = []
        epsilon = 0
        
        
        successful_delivery_and_return = 0
        returned_without_delivery = 0
        delivery_failed_to_return = 0
        failed_deliver_failed_return = 0
        
        
        for episode in range(testing):
            # print("Test Ep.", episode)
            
            if episode in capture_episodes:
                start_recording(episode)
        
        
            congestion.clear()
            generate_dynamic_congestion(episode)
            if episode % 20000 == 0:
                congestion.pop()
                new_congestion = pygame.Rect(280, 300, 100, 100)
                congestion.append(new_congestion)
            path = set()
            drone = (100, 100)
            truck = generate_random_point()
            customer = (100, 700)
            depot = (700, 700)
            emergency = (700, 150) 
            delivered = False
            near_customer = False
            near_depot = False
            near_emergency = False
            target = None
            returned = False
            steps = 0
            viz_training = True
            viz_step=99
            total_reward = 0
            speed = 1
            state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, False, is_near_congestion(drone), (distance(drone, customer) < 50, distance(drone, depot) < 50, distance(drone, emergency) < 50))
            dir = None
            battery = fullBattery
            while not is_within_radius(drone, depot) and not is_within_radius(drone, emergency) and battery > 0:
                if steps % 200 == 0:
                    if steps != 0:
                        random.shuffle(congestion)
                        congestion.pop()
                    add_random_congestion(congestion, 800, 800)
                # epsilon *= 0.9995
                if state not in q_table:
                    q_table[state] = {action: 0 for action in actions}
                action = choose_action(state, epsilon)
                dronePrev = drone
                
                if action == 'move_to_customer':
                    target = customer
                    drone = move_towards(drone, target, speed)
                elif action == 'move_to_depot':
                    target = depot
                    drone = move_towards(drone, target, speed)
                elif action == 'move_to_emergency':
                    target = emergency
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up':
                    target = (drone[0], max(drone[1] - 20, 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down':
                    target = (drone[0], min(drone[1] + 20, height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_left':
                    target = (max(drone[0] - 20, 0), drone[1])
                    drone = move_towards(drone, target, speed)
                elif action == 'move_right':
                    target = (min(drone[0] + 20, width-1), drone[1])
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up_left':
                    target = (max(drone[0] - math.sqrt(20), 0), max(drone[1] - math.sqrt(20), 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_up_right':
                    target = (min(drone[0] + math.sqrt(20), width-1), max(drone[1] - math.sqrt(20), 0))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down_left':
                    target = (max(drone[0] - math.sqrt(20), 0), min(drone[1] + math.sqrt(20), height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'move_down_right':
                    target = (min(drone[0] + math.sqrt(20), width-1), min(drone[1] + math.sqrt(20), height-1))
                    drone = move_towards(drone, target, speed)
                elif action == 'increase_speed':
                    if speed < 20:
                        speed += 1
                    else:
                        speed = 20
                elif action == 'decrease_speed':
                    if speed > 1:
                        speed -= 1
                    else:
                        speed = 1
                elif action == 'move_away':
            
                    target, dir = move_away_from_congestion(state)
                    drone = move_towards(drone, target, speed)
                
                
                
                # Update delivery status if drone reaches customer
            
            
                battery = round(battery - 0.5 * 1 * (speed ** 2)) if battery > 0 else 0
                isBatteryLow = False if battery / fullBattery > 0.25 else True
                # battery = round(battery / 5) * 5
                
                # print(battery)
            
                reward = get_reward(drone, customer, emergency, depot, delivered, truck, speed, near_customer, near_depot, near_emergency, battery)
            
                if battery < 0:
                    battery = 0
    
                        
                        
            
                for obstacle in obstacles:
                    if obstacle.collidepoint(drone):
                        drone = dronePrev
                
                path.add(drone)
                # if action == 'move_to_customer':
                #     print(reward)
                
                if distance(drone, customer) < 50:
                    near_customer = True
            
                if distance(drone, depot) < 50:
                    near_depot = True
            
                if distance(drone, emergency) < 50:
                    near_emergency = True
                
                if not delivered and is_within_radius(drone, customer):
                    # print("delivered")
                    delivered = True
                
            
                total_reward += reward
                next_state = ((int(drone[0] // 20) * 20, int(drone[1] // 20) * 20), delivered, isBatteryLow, is_near_congestion(drone), (near_customer, near_depot, near_emergency))
                if next_state not in q_table:
                    q_table[next_state] = {action: 0 for action in actions}
                # if cheese_eaten:
                #     print(next_state)
                q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
                state = next_state
                # Visualization
                if episode in capture_episodes:
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
            
                    window.fill(WHITE)
            
                    pygame.draw.circle(window, RED, drone, 5)
                    
                    if not delivered:
                        pygame.draw.circle(window, GREEN, customer, 20)
                    pygame.draw.circle(window, BLUE, depot, 20)
                    pygame.draw.circle(window, CYAN, emergency, 20)
                    for cong in congestion:
                        pygame.draw.rect(window, RED, cong)
                    for obstacle in obstacles:
                        pygame.draw.rect(window, ORANGE, obstacle)
                    for p in path:
                        pygame.draw.circle(window, (0, 255, 0), p, 5)
                    # Draw obstacles
                    
            
                   
            
                    if target is not None:
                        pygame.draw.line(window, GREEN, drone, customer, 1)
                        pygame.draw.line(window, BLUE, drone, depot, 1)
                        pygame.draw.line(window, CYAN, drone, emergency, 1)
                    pygame.draw.circle(window, BLACK, truck, 5)
            
                    vx_text = font.render(f'Episode: {episode + 1}', True, BLACK)
            
                    vy_text = font.render(f'Reward: {reward:.2f}', True, BLACK)
                    batter_text = font.render(f'Battery: {battery}', True, BLACK)
                    window.blit(vx_text, (10, 10))
                    window.blit(vy_text, (10, 50))
                    window.blit(batter_text, (10, 100))
                    pygame.display.flip()
                    
                    if episode in capture_episodes:
                        record_frame(episode)
            
                    time.sleep(0.01)
                
                
                
                if delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
                    successful_delivery_and_return += 1
                    returned = True
                    break
                
                
                if not delivered and (is_within_radius(drone, depot) or is_within_radius(drone, emergency)):
                    returned_without_delivery += 1
                    returned = True
                    break
        
                
                
                
                
                
                steps += 1
                
                
            if delivered and not returned:
                delivery_failed_to_return += 1
                
            if not delivered and not returned:
                failed_deliver_failed_return += 1
                
                
                
                
                
            if episode in capture_episodes:
                stop_recording(episode)
                
            if delivered:
                delivered_rewards.append(total_reward)
            else:
                total_rewards.append(total_reward)
            
            # print(total_reward)
            
            plot_reward.append(total_reward)
        
        
        
        
        
        moving_avg_reward = moving_average(plot_reward, window_size)
        # plot_x = range(len(plot_reward))
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(window_size - 1, len(plot_reward)), moving_avg_reward, label="Moving Average", color="orange")
        
        # # Set labels and title
        # plt.xlabel('Episode Number')
        # plt.ylabel('Moving Average Reward')
        # plt.title('Moving Average Reward vs. Episode Number')
        
        # # Add grid and legend
        # plt.legend()
        # plt.grid(True)
        
        # # Display the plot
        # plt.show()
        
        save_plot_reward_to_file(plot_reward)
        
        print("\n")   
        print("Reward Ratio: ", rr, "\tTrust Zone Redius: ", tzr)
        print(f"Successful deliveries and returns: {successful_delivery_and_return}")
        print(f"Returned without delivery: {returned_without_delivery}")
        print(f"Delivers but failed to return: {delivery_failed_to_return}")
        print(f"Both failed: {failed_deliver_failed_return}")
        print("\n")


print(time.time()-start)


pygame.quit()
sys.exit()
