import pygame
import time

class AlarmSystem:
    def __init__(self, sound_path):
        pygame.mixer.init()
        self.last_played = 0
        self.cooldown = 2.0 # Wait time

        try:
            self.alarm_sound = pygame.mixer.Sound(sound_path)
        except pygame.error as e:
            print(f"Couldn't load sound {sound_path}: {e}")
    
    def trigger(self):
        current_time = time.time()
        if current_time - self.last_played > self.cooldown:
            self.alarm_sound.play()
            self.last_played = current_time
            print("FAAAAHHHH")