import pygame
import pygame.camera

print("Initializing camera...")
pygame.camera.init()
cameras = pygame.camera.list_cameras() #Camera detected or not
print(cameras)
print("Done!")
#0, 14, 15
cam = pygame.camera.Camera("/dev/video0", (640, 480))
cam.start()
        
img = cam.get_image()
pygame.image.save(img,"./frame.jpeg")
