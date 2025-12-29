import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import os


DATA_FILENAME = r"C:\Users\mclef\Desktop\thesis\my_working_dataset.csv"

class ProfessionalHybridEnv(gym.Env):
    def __init__(self, df):
        super(ProfessionalHybridEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.soc = 60.0 

        # Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Action Space (Continuous: 0.0 to 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc = 60.0 
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
    #SAFE READ: We use the cleaned column names 
    #If the column is missing for any reason, we default to 0 to avoid crashes.
        eng_pwr = row.get('Engine Power (kW)', 0)
        reg_pwr = row.get('Regenerative Braking Power (kW)', 0)
        
        power_demand = eng_pwr - reg_pwr

        obs = np.array([
            row.get('Speed (km/h)', 0),
            row.get('Acceleration (m/s²)', 0),
            power_demand,
            self.soc
        ], dtype=np.float32)
        return obs

    def step(self, action):
        u_engine = float(action[0])
        row = self.df.iloc[self.current_step]
        
        eng_pwr = row.get('Engine Power (kW)', 0)
        reg_pwr = row.get('Regenerative Braking Power (kW)', 0)
        power_demand = eng_pwr - reg_pwr
        
        engine_power = 0.0
        battery_power = 0.0
        fuel_consumption = 0.0
        
        if power_demand <= 0:
            battery_power = power_demand
            self.soc -= (battery_power * 0.7) / 100 
        else:
            engine_power = power_demand * u_engine
            battery_power = power_demand * (1.0 - u_engine)
            
            if engine_power > 0:
                fuel_consumption = (engine_power * 0.00025) 
            
            self.soc -= (battery_power * 0.05) 
            
        self.soc = np.clip(self.soc, 0, 100)
        
        reward = 0
        reward -= fuel_consumption * 100 
        if self.soc < 30: reward -= 50 * (30 - self.soc)
        elif self.soc > 90: reward -= 50 * (self.soc - 90)
            
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {"fuel": fuel_consumption, "soc": self.soc}
        
        return self._get_obs(), reward, terminated, truncated, info

if __name__ == "__main__":
    if not os.path.exists(DATA_FILENAME):
        print(f"Error cannot find the file {DATA_FILENAME}")
    else:
        print("Loading data...")
        df = pd.read_csv(DATA_FILENAME)
        
        #Extra Cleaning
        df.columns = df.columns.str.strip()
        print(f"Columns after cleaning: {df.columns.tolist()}")
        
        #Checking
        if 'Regenerative Braking Power (kW)' not in df.columns:
            print(" The column is still missing! Something is wrong with the file.")
            # Create the column if it is missing, to ensure the script can run.
            df['Regenerative Braking Power (kW)'] = 0.0
            print("A temporary empty Regen column was created so we can continue")
        
        env = ProfessionalHybridEnv(df)
        
        print("Starting to train AI Agent (PPO)...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        model.learn(total_timesteps=50000)
        
        print("Training completed!")
        model.save("final_hybrid_agent_model")
        
        print("\nTEST DRIVE REPORT")
        obs, _ = env.reset()
        total_ai_fuel = 0
        for _ in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            total_ai_fuel += info['fuel']
            if done: break
        
        print(f"AI Consumption (1000 steps): {total_ai_fuel:.2f} Liters")