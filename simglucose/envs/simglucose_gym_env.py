from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pandas as pd
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename('simglucose', 'params/vpatient_params.csv')


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name='adult#001', custom_scenario='random', risk_metric='clarke', reward_fun=None, start_time=None, normalise_state_space=True, normalise_action_space=True, weight_state=False, age_state=False, height_state=False, tdi_state=False, meal_state=False, pid_state=False, n_hours=4, insulin_max=1, scale_factor=0.1):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        self.np_random, _ = seeding.np_random()
        self.patient_name = patient_name
        self.custom_scenario = custom_scenario
        self.risk_metric = risk_metric
        self.reward_fun = reward_fun
        self.normalise_state_space = normalise_state_space
        self.normalise_action_space = normalise_action_space
        self.weight_state = weight_state
        self.age_state = age_state
        self.height_state = height_state
        self.tdi_state = tdi_state
        self.meal_state = meal_state
        self.pid_state = pid_state
        self.insulin_max = insulin_max
        self.scale_factor = scale_factor

        self.fixed_start_time = True if start_time else False
        self.start_time = start_time if self.fixed_start_time else datetime(2018,1,1,self.np_random.randint(0,24),0,0)
        self.weight = pd.read_csv('simglucose/params/vpatient_params.csv').query('Name=="{}"'.format(self.patient_name))['BW'].item()
        self.height = pd.read_csv('simglucose/params/Quest.csv').query('Name=="{}"'.format(self.patient_name))['Height'].item()
        self.age = pd.read_csv('simglucose/params/Quest.csv').query('Name=="{}"'.format(self.patient_name))['Age'].item()
        self.tdi = pd.read_csv('simglucose/params/Quest.csv').query('Name=="{}"'.format(self.patient_name))['TDI'].item()      
        self.env, _, _, _ = self._create_env_from_random_state()
        self.state_length = int((n_hours * 60) / self.env.sample_time)

        self.min_bg = 0
        self.max_bg = 400
        self.min_insulin = self.action_space.low
        self.max_insulin = self.action_space.high
        self.min_meal = 0
        self.max_meal = 50

        self.min_weight = 20
        self.max_weight = 120
        self.min_age = 5
        self.max_age = 70
        self.min_height = 120
        self.max_height = 180
        self.min_tdi = 15
        self.max_tdi = 70

    def normalise(self, x, new_min, new_max, old_min, old_max):
        a = (new_max - new_min)/(old_max - old_min)
        b = x - old_min
        c = new_min
        return a*b + c

    def get_state(self):
        # Get history w.r.t n_hours for the state length
        bg = self.env.CGM_hist[-self.state_length:]
        insulin = self.env.insulin_hist[-self.state_length:]
        meal = self.env.CHO_hist[-self.state_length:]

        # Norm individual state components to between 0 and 1 
        # Except bg which can be outside of this range if bg is outside of TIR range
        if self.normalise_state_space:
            bg = self.normalise(np.array(bg), 0, 1, self.min_bg, self.max_bg)
            insulin = self.normalise(np.array(insulin), 0, 1, self.min_insulin, self.max_insulin)
            meal = self.normalise(np.array(meal), 0, 1, self.min_meal, self.max_meal)
            weight = self.normalise(np.array([self.weight]), 0, 1, self.min_weight, self.max_weight)
            age = self.normalise(np.array([self.age]), 0, 1, self.min_age, self.max_age)
            height = self.normalise(np.array([self.height]), 0, 1, self.min_height, self.max_height)
            tdi = self.normalise(np.array([self.tdi]), 0, 1, self.min_tdi, self.max_tdi)

        # Fill missing state values with -1
        if len(bg) < self.state_length:
            bg = np.concatenate((np.full(self.state_length - len(bg), -1), bg))
        if len(insulin) < self.state_length:
            insulin = np.concatenate((np.full(self.state_length - len(insulin), -1), insulin))
        if len(meal) < self.state_length:
            meal = np.concatenate((np.full(self.state_length - len(meal), -1), meal))

        # Create state space array
        # TODO: Add TDI or BMR to state space as a metabolic parameter
        return_arr = [bg, insulin]
        if self.meal_state: return_arr.append(meal)
        if self.weight_state: return_arr.append(weight)
        if self.age_state: return_arr.append(age)
        if self.height_state: return_arr.append(height)
        if self.tdi_state: return_arr.append(tdi)
        return_arr = np.concatenate(return_arr)

        # Just return last bg reading for PID controller (and sample time)
        if self.pid_state: return_arr = np.array([bg[-1], self.env.sample_time])

        return return_arr

    def step(self, action):
        if type(action) is np.ndarray:
            action = action.item()

        if self.normalise_action_space: 
            action = self.normalise(action, -self.scale_factor*self.insulin_max, self.insulin_max, -1, 1)
        act = Action(basal=action, bolus=0)

        _, reward, done, info = self.env.step(act, self.reward_fun)

        state = self.get_state()
            
        return state, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        return self.get_state()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Seed gets passed as a uint, but gets checked as an int elsewhere, so we need to keep it below 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        self.start_time = self.start_time if self.fixed_start_time else datetime(2018,1,1,self.np_random.randint(0,24),0,0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed2)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed3)
        scenario = RandomScenario(self.start_time, self.weight, self.height, self.age, self.patient_name[:-4], custom_scenario=self.custom_scenario, seed=seed4)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario, self.risk_metric)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    def show_history(self):
        return self.env.show_history()

    @property
    def env_time(self):
        return self.env.time

    @property
    def action_space(self):
        return spaces.Box(low=0, high=self.insulin_max, shape=(1,))

    @property
    def observation_space(self):
        st = self.get_state()
        return spaces.Box(low=0, high=np.inf, shape=(len(st),))