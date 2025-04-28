from datetime import datetime, timedelta
import pandas as pd
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from queue import PriorityQueue

import torch
import numpy as np
# 设置随机种子以确保结果可重现
random.seed(42)
torch.manual_seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ElectricVehicle:
    def __init__(self, id, connection_time, done_charging_time, kwh_delivered,
                 charging_power, battery_capacity, max_queue_length=10):
        self.id = id
        self.connection_time = connection_time
        self.done_charging_time = done_charging_time
        self.kwh_delivered = kwh_delivered
        self.charging_power = charging_power
        self.battery_capacity = battery_capacity   #电池容量
        self.current_battery = random.uniform(0.2, 0.6) * battery_capacity    #当前电池状态
        # self.charging_time = 0    #由所选充电站的功率及当前电池状态计算出的充电时长
        # self.required_charge = 0    #充电需求
        self.charging_pile = None    #充电时所在充电站中的充电桩的位置
        # self.ve_xy = torch.rand(1, 2)    #当前所在位置
        # self.max_distance = max_distance    #能接受的最远距离
        self.max_price = np.clip(np.random.normal(0.7, 0.2), 0.7, 1)    #能接受的最高电价
        self.max_queue_length = max_queue_length    #能接受的最长排队数
        self.cs_idx = None    #所在充电站的序号
        self.personality = np.random.normal(1, 0.3)  #车主性格（服从正态分布）
        self.charge_tag = None   #是否选择充电

    #根据电池电量计算紧急程度
    def calculate_emergency(self):
        charging_emergency = np.exp(-2 * self.current_battery / self.battery_capacity)    #紧急程度
        return charging_emergency

    #计算权重
    def calculate_weights(self, w_q):  #w_q为等待区长度的权重，未选择充电的车主和在等待区等待充电的车主对等待区长度的考虑不同


        # 根据充电欲望和紧急程度计算权重
        # distance_weight = 1 - charging_desire  # 距离权重：欲望低的车主更看重价格
        queue_time_weight = 0.5 + w_q * self.calculate_emergency()  # 价格权重：紧急程度高的车主更看重距离
        price_weight = 1 - queue_time_weight  # 排队时长权重：紧急程度高的车主更看重排队时长

        return price_weight, queue_time_weight

    #查询最优充电站
    def quary(self, n_cs, cs_price, cs_waitting_queue, w_q):
        best_cs_idx = 0
        max_score = 0
        price_weight, queue_time_weight = self.calculate_weights(w_q)
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

    #选择充电或不充电
    def make_charging_decision(self, current_price, waitting_rate, w_q):

        # weight_emergency = 0.6  # 紧急程度的权重
        weight_price, weight_waitting = self.calculate_weights(w_q)  # 电价、等待时间的权重

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


    #计算充电时长
    # def calculate_charging_time(self, charging_rate):
    #     self.required_charge = self.battery_capacity - self.current_battery
    #     charging_time = self.required_charge / charging_rate[self.cs_idx]
    #     return charging_time

class EVS:
    def __init__(self, data_path: str):
        """
        初始化充电系统
        Args:
            data_path: CSV数据文件路径
        """
        self.data_path = data_path
        self.evs: Dict[str, EV] = {}  # 存储所有车辆
        self.load_data()

    def load_data(self):
        """加载并解析CSV数据"""
        df = pd.read_csv(self.data_path)

        for _, row in df.iterrows():
            # 将时间字符串转换为datetime对象
            connection_time = datetime.strptime(row['connectionTime'], '%H:%M:%S').time()
            done_charging_time = datetime.strptime(row['doneChargingTime'], '%H:%M:%S').time()

            # 创建EV实例
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
        获取指定时间应该到达的车辆
        Args:
            current_time: 当前时间
        Returns:
            返回在指定时间范围内到达的车辆列表
        """
        # 计算当前时间加 15 分钟后的时间
        today = datetime.today()
        current_time = datetime.combine(today, current_time)
        future_time = (current_time + timedelta(minutes=15)).time()
        current_time = current_time.time()

        # 返回符合条件的车辆：连接时间小于等于当前时间，且大于等于当前时间+15分钟
        return [ev for ev in self.evs.values() if current_time <= ev.connection_time <= future_time]

    def remove_ev(self, ev):
        """
        从系统中移除车辆（开始充电后）
        Args:
            ev_id: 车辆ID
        Returns:
            bool: 是否成功移除
        """
        if ev.id in self.evs:
            del self.evs[ev.id]
            return True
        return False



