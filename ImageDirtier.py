import cv2
import numpy as np
class ImageDirtier():
  def __init__(self,number_of_spots,minimal_size,maximal_size):
    self.number_of_spots = number_of_spots
    self.minimal_size = minimal_size
    self.maximal_size = maximal_size
    self.minimal_transparency = 0.8
    self.maximal_transparency = 0.9
    self.lod = 7

  def to_radians(self,angle):
    return angle * np.pi / 180.0

  def point_on_circle(self,center_point,angle,radius):
    x = center_point[0] + radius * np.cos(self.to_radians(angle))
    y = center_point[1] + radius * np.sin(self.to_radians(angle))
    return np.array([x,y])

  def minmax(self,number,limit):
    return max(0,min(number,limit))
  
  def generate_polygons(self,width,height):
    polygons = np.zeros((self.number_of_spots, self.lod, 3))
    for i in range(self.number_of_spots):
      angle = 0
      move_angle = 360 / self.lod
      main_radius = np.random.uniform(self.minimal_size, self.maximal_size)
      center_point = np.random.randint([0 + main_radius, 0 + main_radius],[width - main_radius, height - main_radius])
      bonus = 0
      for j in range(self.lod):
        move = np.random.randint(0,move_angle + bonus)
        diff_radius = np.random.uniform(main_radius / 6,main_radius / 2)
        bonus += move_angle - move
        angle += move
        new_point = self.point_on_circle(center_point, angle, main_radius - diff_radius)
        polygons[i][j] = np.array([self.minmax(new_point[0], width),self.minmax(new_point[1], height), 1],dtype = np.int)
    return polygons
  
  def parse_polygons(self,polygons):
    polygons_contours = []
    for i in range(polygons.shape[0]):
      polygons_contours.append([])
      transformed_polygon = []
      for j in range(polygons.shape[1]):
        transformed_polygon.append([polygons[i][j][0], polygons[i][j][1]])
      polygons_contours[i].append(np.array(transformed_polygon, dtype = np.int))
    return polygons_contours

  def apply_spots(self,image, contours):
    mask = np.zeros(image.shape[:2])
    img2 = np.copy(image)
    for i, contour in enumerate(contours):
      cv2.drawContours(mask, contour, 0, 255*(i+1), -1)
      color = np.random.uniform(0,16)
      cv2.drawContours(img2, contour, 0, (color, color, color), -1)
    transparency = np.random.uniform(self.minimal_transparency, self.maximal_transparency)
    img2 = cv2.addWeighted(image, transparency, img2, 1 - transparency, 0, img2)
    return img2,mask/255.
  
  def apply(self,image):
    polygons = self.generate_polygons(image.shape[0], image.shape[1])
    polygons_contours = self.parse_polygons(polygons)
    return self.apply_spots(image, polygons_contours)