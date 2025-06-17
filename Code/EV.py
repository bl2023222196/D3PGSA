from datetime import datetime, timedelta
import pandas as pd
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from queue import PriorityQueue

import torch
import numpy as np

random.seed(42)
torch.manual_seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ElectricVehicle:
    def __init__(self, id, connection_time, done_charging_time, kwh_delivered,
                 charging_power, battery_capacity, max_queue_length=10):
        self.id = id
        self.connection_time = connection_time
        self.done_charging_time = done_charging_time
        self.kwh_delivered = kwh_delivered
        self.charging_power = charging_power
        self.battery_capacity = battery_capacity   
        self.current_battery = random.uniform(0.2, 0.6) * battery_capacity   
        # self.charging_time = 0    
        # self.required_charge = 0   
        self.charging_pile = None  
        # self.ve_xy = torch.rand(1, 2)   
        # self.max_distance = max_distance   
        self.max_price = np.clip(np.random.normal(0.7, 0.2), 0.7, 1)    
        self.max_queue_length = max_queue_length   
        self.cs_idx = None   
        self.personality = np.random.normal(1, 0.3) 
        self.charge_tag = None 

    
    def calculate_emergency(self):
        charging_emergency = np.exp(-2 * self.current_battery / self.battery_capacity)   
        return charging_emergency

 
    def calculate_weights(self, beta): 
        
        queue_time_weight = 0.5 + beta * self.calculate_emergency()  
        price_weight = 1 - queue_time_weight  
        return price_weight, queue_time_weight

   
    def quary(self, n_cs, cs_price, cs_waitting_queue, beta):
        best_cs_idx = 0
        max_score = 0
        price_weight, queue_time_weight = self.calculate_weights(beta)
        for i in range(n_cs):
            if cs_price[i] > self.max_price:
                continue
            # distance = eucli(self.ve_xy, cs_xy[i])
            # distance_score = 1 - (distance / self.max_distance)
            price_score = 1 - (cs_price[i] / self.max_price)
            queue_score = 1 - (cs_waitting_queue[i].size() / self.max_queue_length)
            total_score =  + price_score * price_weight + queue_score * queue_time_weight
            if max_score < total_score:
                max_score = total_score
                best_cs_idx = i
        return best_cs_idx

 
    def make_charging_decision(self, current_price, waitting_rate, beta):

        
        weight_price, weight_waitting = self.calculate_weights(beta) 

        total_weight = weight_waitting + weight_price
        weight_waitting /= total_weight
        weight_price /= total_weight

        combined_score = (weight_waitting * (1 - waitting_rate[self.cs_idx])) + (weight_price * (self.max_price - current_price))

        decision_threshold = 0.4
        # print(combined_score * charge_desire)
        if current_price > 3:
            self.charge_tag = False
        elif combined_score * self.personality > decision_threshold:
            self.charge_tag = True
        else:
            self.charge_tag = False


 

class EVS:
    def __init__(self, data_path: str):
        """
        
        Args:
            data_path: CSV数据文件路径
        """
        self.data_path = data_path
        self.evs: Dict[str, EV] = {}  
        self.load_data()

    def load_data(self):
       
        df = pd.read_csv(self.data_path)

        for _, row in df.iterrows():
          
            connection_time = datetime.strptime(row['connectionTime'], '%H:%M:%S').time()
            done_charging_time = datetime.strptime(row['doneChargingTime'], '%H:%M:%S').time()

           
            ev = ElectricVehicle(
                id=row['_id'],
                connection_time=connection_time,
                done_charging_time=done_charging_time,
                kwh_delivered=row['kWhDelivered'],
                charging_power=row['chargingPower'],
                battery_capacity=65
            )
            self.evs[ev.id] = ev

    def postpone_ev(self, ev, multiplier: int = 1):
        ev.postpone_charging(multiplier)
        return None

    def get_evs_at_time(self, current_time) -> List[ElectricVehicle]:
        """
        Args:
            current_time: 当前时间
        """
        
        today = datetime.today()
        current_time = datetime.combine(today, current_time)
        future_time = (current_time + timedelta(minutes=15)).time()
        current_time = current_time.time()

       
        return [ev for ev in self.evs.values() if current_time <= ev.connection_time <= future_time]

    def remove_ev(self, ev):
        """
       
        Args:
            ev_id: 车辆ID
        Returns:
           
        """
        if ev.id in self.evs:
            del self.evs[ev.id]
            return True
        return False



