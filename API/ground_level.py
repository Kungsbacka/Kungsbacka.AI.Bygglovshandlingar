import cv2
import numpy as np
import numpy as np
import math
import response_model as rm
import cv2 
import logging
class GroundLevel:
    def __init__(self):
        pass
    def find_lines_exiting_boxes(self, image, boxes):
        """Find lines exiting the bounding boxes
        Args:
            image (np.array): Image to find lines in
            boxes (list): List of bounding boxes
        Returns:
            list: List of lines exiting the bounding boxes
        """
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            logging.error("No lines found")
            return None
        
        # Define a function to check if a point lies within a bounding box
        def point_inside_box(point, box):
            """Check if a point lies within a bounding box"""
            x, y = point
            x1, y1, x2, y2 = box
            return x1-20 <= x <= x2 + 20 and y1 -20 <= y <= y2+20
        
        # Iterate over the lines and check for intersections with bounding boxes
        lines_exiting_boxes = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            for box in boxes:
                x1_box, y1_box, x2_box, y2_box = box
                if (point_inside_box((x1, y1), box) and not point_inside_box((x2, y2), box)) or \
                (point_inside_box((x2, y2), box) and not point_inside_box((x1, y1), box)):
                    if self.is_same_direction(line[0], box, 45):
                        lines_exiting_boxes.append(line)
                        break  # Move to the next line if this one intersects with any box
        
        return lines_exiting_boxes

    def filter_lines_by_distance(self, lines, distance_threshold):
                filtered_lines = []
                for i in range(len(lines)):
                    x1, y1, x2, y2 = lines[i][0]
                    for j in range(i + 1, len(lines)):
                        x3, y3, x4, y4 = lines[j][0]
                        distance = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
                        if distance > distance_threshold:
                            filtered_lines.append(lines[i])
                            break
                return filtered_lines


    def distance(self, x1, y1, x2, y2):
        # Calculate Euclidean distance between two points
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def point_line_distance(self, x, y, x1, y1, x2, y2):
        # Calculate the perpendicular distance from a point (x, y) to a line defined by two points (x1, y1) and (x2, y2)
        return abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / self.distance(x1, y1, x2, y2)

    def check_line_near_corner(self, line, box, radius):
        x1_line, y1_line, x2_line, y2_line = line
        x1_box, y1_box, x2_box, y2_box = box
        
        # Check if the line is entirely outside the bounding box
        if (x1_line < x1_box and x2_line < x1_box) or (x1_line > x2_box and x2_line > x2_box) or \
        (y1_line < y1_box and y2_line < y1_box) or (y1_line > y2_box and y2_line > y2_box):
            return False
        
        # Calculate the four corner points of the box
        corners = [(x1_box, y1_box), (x2_box, y1_box), (x1_box, y2_box), (x2_box, y2_box)]
        
        # Check if any point on the line is within the radius of any corner of the box
        for corner in corners:
            x_corner, y_corner = corner
            if self.point_line_distance(x_corner, y_corner, x1_line, y1_line, x2_line, y2_line) <= radius:
                return True
        
        return False

    def find_lines_exiting_short_sides(self, image, boxes):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection using Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            logging.error("No lines found")
            return None
        
        # Define a function to check if a point lies near the short sides of a bounding box
        def near_short_sides(self, point, box, distance_threshold):
            x, y = point
            x1, y1, x2, y2 = box
            # Check if the point is within the distance threshold from the short sides
            return (x1 <= x <= x1 + distance_threshold or x2 - distance_threshold <= x <= x2) \
                and (y1 <= y <= y1 + distance_threshold or y2 - distance_threshold <= y <= y2)
        
        # Set a distance threshold for the short sides
        short_side_distance_threshold = 20  # Adjust as needed
        
        # Iterate over the lines and check for intersections with short sides of bounding boxes
        lines_exiting_short_sides = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            for box in boxes:
                x1_box, y1_box, x2_box, y2_box = box
                if (near_short_sides((x1, y1), box, short_side_distance_threshold) and \
                    not near_short_sides((x2, y2), box, short_side_distance_threshold)) or \
                (near_short_sides((x2, y2), box, short_side_distance_threshold) and \
                    not near_short_sides((x1, y1), box, short_side_distance_threshold)):
                    lines_exiting_short_sides.append(line)
                    break  # Move to the next line if this one intersects with any short side
        
        return lines_exiting_short_sides

    def filter_lines_by_direction(self, lines, boxes, max_angle_difference):
        """Filter lines based on direction
        Args:
            lines (list): List of lines
            boxes (list): List of bounding boxes
            max_angle_difference (float): Maximum angle difference in degrees
            Returns:
            list: List of lines that match the direction of the boxes
        """
        ok_lines_index = set()
        for i, line in enumerate(lines):
            for box in boxes:
                x1_box, y1_box, x2_box, y2_box = box
                line1 = [x1_box, y1_box, x1_box, y2_box]
                line2 = [x1_box, y1_box, x2_box, y1_box]
                
                if self.line_length(line1) > self.line_length(line2):
                    longest_line = line1
                else:
                    longest_line = line2
            
            
                if self.angle_difference(longest_line, line[0]) < max_angle_difference:
                    ok_lines_index.add(i)
                    break

        
        return [lines[i] for i in ok_lines_index]

    def is_same_direction(self, line, box, max_angle_difference):
        """Check if a line is in the same direction as a box
        Args:
            line (list): Line coordinates [x1, y1, x2, y2]
            box (list): Box coordinates [x1, y1, x2, y2]
            max_angle_difference (float): Maximum angle difference in degrees
        Returns:
            bool: True if the line is in the same direction as the box, False otherwise
        """

        x1_box, y1_box, x2_box, y2_box = box
        line1 = [x1_box, y1_box, x1_box, y2_box]
        line2 = [x1_box, y1_box, x2_box, y1_box]
        
        if self.line_length(line1) > self.line_length(line2):
            longest_line = line1
        else:
            longest_line = line2


        if self.angle_difference(longest_line, line) < max_angle_difference:
            return True
        return False
            

            

    def calculate_angle(self, line):
        """Calculate the angle of a line
        Args:
            line (list): Line coordinates [x1, y1, x2, y2]
        Returns:
            float: Angle of the line in degrees
            """

        x1, y1, x2, y2 = line
        # Calculate the angle of a line using the arctangent function
        dx = x2 - x1
        dy = y2 - y1
        return math.atan2(dy, dx) * 180 / math.pi

    def angle_difference(self, line1, line2):
        """Calculate the difference in angle between two lines
        Args:
            line1 (list): Line coordinates [x1, y1, x2, y2]
            line2 (list): Line coordinates [x1, y1, x2, y2]
        Returns:
            float: Absolute difference in angle between the two lines in degrees
        """
        # Calculate the angle of each line
        angle1 = self.calculate_angle(line1)
        angle2 = self.calculate_angle(line2)
        
        # Calculate the absolute difference between the angles
        diff = abs(angle1 - angle2)
        
        # Ensure the difference is between 0 and 180 degrees
        if diff > 180:
            diff = 360 - diff
        
        return diff

    def line_length(self, line):
        """Calculate the length of a line
        Args:
            line (list): Line coordinates [x1, y1, x2, y2]
        Returns:
            float: Length of the line
        """
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def validate_ground_lines(self, image, boxes) -> rm.ComponentResponse:
        """Validate ground lines in image
        Args:
            image (np.array): Image to validate
            boxes (list): List of bounding boxes
        Returns:
            rm.ComponentResponse: Response with status, code and message
        """
        # Load the image and bounding boxes
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        

        # Find lines exiting the boxes
        exiting_lines = self.find_lines_exiting_boxes(image, boxes)

        # Filter the lines based on direction
        exiting_lines = self.filter_lines_by_direction(exiting_lines, boxes, 45)


        final_lines = []
        boxes_with_lines = set()
        for i, box in enumerate(boxes):
            for line in exiting_lines:
                if self.check_line_near_corner(line[0], box, radius=max(image.shape[0]//50, image.shape[1]//50)):
                    final_lines.append(line)
                    boxes_with_lines.add(i)


        #final_lines = filter_lines_by_distance(final_lines, 50)


        logging.info(f"Ground lines found: {len(final_lines)}")
        logging.info(f"Boxes: {len(boxes)}")
        logging.info(f"Boxes with ground lines: {len(boxes_with_lines)}")


        if len(boxes) == len(boxes_with_lines):
            logging.info("Success: Ground lines found for all sides")
            return {"status":True, "code":"300", "msg":"Success"}
        elif len(boxes_with_lines) == 0:
            logging.error("Error: No ground lines found")
            return {"status":False, "code":"310", "msg":"No ground lines found"}
        elif len(boxes) > len(boxes_with_lines):
            logging.error("Error: Ground lines missing for at least one side")
            return {"status":False, "code":"340", "msg":"Ground Line missing or cant be detected for at least one side"}
        

