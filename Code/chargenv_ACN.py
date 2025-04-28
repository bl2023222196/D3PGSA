import random
import numpy as np
from collections import deque
import torch
import csv
import EV
from datetime import datetime, timedelta, time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_price = 1

# 计算欧氏距离
def eucli(a, b):
    dist = torch.sqrt(torch.sum((a - b) ** 2))
    return dist

#充电站的等待队列
class WaittingQueue:
    def __init__(self, max_length):
        self.queue = []
        self.max_length = max_length
    def enqueue(self, item):
        if len(self.queue) < self.max_length:
            self.queue.append(item)
        # else:
        #     print("Queue is full. Cannot enqueue more items.")
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            raise IndexError("Queue is empty")
    def is_empty(self):
        return len(self.queue) == 0
    def is_full(self):
        if len(self.queue) == self.max_length:
            return True
        else:
            return False
    def size(self):
        return len(self.queue)


#电动车辆



class Env:
    def __init__(self, current_time_step, seed,  initial_price, vehicle_arrival_rate, path,
                 num_piles=30, charging_rate=50, WaittingQueue_maxlen=50, n_cs=1): # charging_rate=50kw
        self.n_cs = n_cs   #充电站数量
        self.initial_price = initial_price   #初始价格
        self.price = torch.full((self.n_cs,1), self.initial_price,dtype=torch.float32, device=device)  # 充电站当前价格(5,1)
        self.num_piles = torch.full((self.n_cs,1), num_piles, device=device)  # 充电站的充电桩数量(5,1)
        self.occupied_piles = [[] for _ in range(self.n_cs)]  # 每个充电站已被占用的充电桩序号[[] [] [] [] []]
        self.occupied_rate = torch.zeros((self.n_cs, 1), device=device)     #充电桩占用率
        self.cs_revenue = torch.zeros((self.n_cs, 1), device=device)    # 每个充电站的利润(5,1)
        self.charging_rate = torch.full((self.n_cs,1), charging_rate, device=device)   #充电功率
        self.charging_vehicles = [[] for _ in range(self.n_cs)]  # 每个充电站正在充电的车辆[[] [] [] [] []]
        self.waittingQueue_maxlen = WaittingQueue_maxlen
        self.waitting_queue = [WaittingQueue(self.waittingQueue_maxlen) for i in range(self.n_cs)]
        self.waitting_rate = torch.zeros((self.n_cs, 1), device=device)
        self.Max_KW = torch.tensor(1000, device=device)
        self.current_load = torch.zeros((self.n_cs, 1), device=device)  
        # self.cs_xy = torch.rand((n_cs, 1, 2))
        # self.current_time_step = current_time_step
        self.current_time = torch.tensor(0, device=device)
        self.vehicle_arrival_rate = vehicle_arrival_rate
        self.current_step_revenue = torch.zeros((self.n_cs, 1), dtype=torch.float32, device=device)
        self.total_revenue = 0
        # self.ev = []
        self.seed = seed
        self.start_datetime = time(0, 0, 0)  # 模拟开始时间
        self.current_datetime = self.start_datetime  # 当前模拟时间
        self.evs = EV.EVS(path)
        self.done = None
        self.data_path = path
    #开始充电
    def charge_vehicles_start(self, vehicle):
        available_pile = self.find_available_pile(vehicle.cs_idx)
        if available_pile is not None and self.Max_KW - self.current_load >= vehicle.charging_power:
            self.occupied_piles[vehicle.cs_idx].append(available_pile)
            vehicle.charging_pile = available_pile
            self.charging_vehicles[vehicle.cs_idx].append(vehicle)
            self.current_load[vehicle.cs_idx] += vehicle.charging_power
        else:
            self.waitting_queue[vehicle.cs_idx].enqueue(vehicle)
            
    #完成充电
    def charge_vehicles_end(self, vehicle):
        previous_env_time = (datetime.combine(datetime.today(), self.current_datetime) - timedelta(minutes=15)).time()

        today = datetime.today()
        done_charging_datetime = datetime.combine(today, vehicle.done_charging_time)
        previous_env_datetime = datetime.combine(today, previous_env_time)

        # 计算实际充电时间（从上个时间点到充电完成）
        charging_hours = (done_charging_datetime - previous_env_datetime).total_seconds() / 3600

        # 确保充电时间是正数且不超过15分钟（0.25小时）
        charging_hours = max(0, min(charging_hours, 0.25))

        # 转换为tensor并计算收益
        charging_hours_tensor = torch.tensor(charging_hours, device=device)
        self.current_step_revenue += charging_hours_tensor * min(self.price, torch.tensor(vehicle.max_price, device=device)) * vehicle.charging_power
        self.occupied_piles[vehicle.cs_idx].remove(vehicle.charging_pile)
        
    #寻找空闲充电桩
    def find_available_pile(self, cs_idx):
        available_piles = list(range(self.num_piles[cs_idx]))
        occupied_piles = list(self.occupied_piles[cs_idx])
        for pile in occupied_piles:
            available_piles.remove(pile)
        if available_piles:
            return random.choice(available_piles)
        return None
    # def update_price(self, action):
    #     for i in range(len(self.price)):
    #         self.price[i] += action[i]
    #         if self.price[i]>1.2:
    #             self.price[i] = 1.2
    
    def reset(self):
        # with open('num_ev10.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for item in self.ev:
        #         writer.writerow([item])
        self.price = torch.full((self.n_cs, 1), self.initial_price, dtype=torch.float32, device=device)
        self.occupied_piles = [[] for _ in range(self.n_cs)]  # [[] [] [] [] []]
        self.occupied_rate = torch.zeros((self.n_cs, 1), device=device)
        self.cs_revenue = torch.zeros((self.n_cs, 1), device=device)  # (5,1)
        self.charging_vehicles = [[] for _ in range(self.n_cs)]  # [[] [] [] [] []]
        self.waitting_queue = [WaittingQueue(self.waittingQueue_maxlen) for i in range(self.n_cs)]
        self.waitting_rate = torch.zeros((self.n_cs, 1), device=device)
        self.start_datetime = time(11, 0, 0)  # 模拟开始时间
        self.current_datetime = self.start_datetime  # 当前模拟时间
        self.current_load = torch.zeros((self.n_cs, 1), device=device)  
        self.total_revenue = 0
        self.done = False
        self.current_time = torch.tensor(0, device=device)
        self.evs = EV.EVS(self.data_path)
        self.current_step_revenue = torch.zeros((self.n_cs, 1), dtype=torch.float32, device=device)
        # self.ev = []
        np.random.seed(self.seed)
        state = torch.cat((self.price, self.occupied_rate,
                           self.waitting_rate, self.current_time.view(1, 1)), dim=0)
        state = torch.transpose(state, 0, 1)
        # return state.unsqueeze(0)
        return state
    # def update_price(self, action):
    #     self.price += action
    #     if self.price < 0:
    #         self.price = 0
    def step(self,action):

        self.current_step_revenue = torch.zeros((self.n_cs, 1), dtype=torch.float32, device=device)
        for i in range(1, 5):
            for j in range(len(self.charging_vehicles)):  # 遍历每个充电站
                if len(self.charging_vehicles[j]) > 0:  # 遍历该充电站种所有在充电的车
                    for charging_vehicle in reversed(self.charging_vehicles[j]):
                        if charging_vehicle.done_charging_time <= self.current_datetime:
                            self.charge_vehicles_end(charging_vehicle)
                            self.charging_vehicles[j].remove(charging_vehicle)
                        else:
                            self.current_step_revenue += torch.tensor(0.25, device=device) * min(self.price, torch.tensor(charging_vehicle.max_price, device=device)) * charging_vehicle.charging_power

            for k in range(len(self.charging_vehicles)):
                if len(self.occupied_piles[k]) < self.num_piles[k] and self.waitting_queue[k].size() > 0:
                    for _ in self.waitting_queue[k].queue:
                        vehicle = self.waitting_queue[k].dequeue()
                        vehicle.make_charging_decision(self.price, self.waitting_rate, 0.1)
                        if vehicle.charge_tag is False:
                            continue
                        else:
                            self.charge_vehicles_start(vehicle)
                        # self.charging_vehicles[k].append(vehicle)

            if i == 1:
                self.price += action
                if self.price.item() < 0.1:
                    self.price = torch.full((self.n_cs, 1), 0.1, device=device)

            arriving_evs = self.evs.get_evs_at_time(self.current_datetime)
            for vehicle in arriving_evs:
                vehicle.cs_idx = vehicle.quary(self.n_cs, self.price, self.waitting_queue, 0.3)
                cs_idx = vehicle.cs_idx
                vehicle.make_charging_decision(self.price, self.waitting_rate, 0.3)
                if vehicle.charge_tag is False:
                    continue
                else:
                    if len(self.occupied_piles[cs_idx]) < self.num_piles[cs_idx]:
                        self.charge_vehicles_start(vehicle)
                    else:
                        self.waitting_queue[cs_idx].enqueue(vehicle)
                    self.evs.remove_ev(vehicle)

            # 处理上一步到达的车辆


            self.current_datetime = (
                    datetime.combine(datetime.today(), self.current_datetime) + timedelta(minutes=15)).time()



        #更新每个充电站的使用情况


        for i in range(self.n_cs):
            self.occupied_rate[i] = len(self.occupied_piles[i]) / self.num_piles[i]
            self.waitting_rate[i] = self.waitting_queue[i].size() / self.waittingQueue_maxlen
        self.total_revenue += self.current_step_revenue
        next_state = torch.cat((self.price, self.occupied_rate,
                                self.waitting_rate, self.current_time.view(1, 1)), dim=0)
        next_state = torch.transpose(next_state, 0, 1)
        reward = self.current_step_revenue
        # return next_state.unsqueeze(0), reward, done
        self.current_time += 1
        if self.current_time == 24:
            self.done = True

        return next_state, reward, self.done

