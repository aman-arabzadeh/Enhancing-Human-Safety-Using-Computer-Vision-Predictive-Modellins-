

"""
Author: Koray Aman Arabzadeh
Thesis: Mid Sweden University.
Bachelor Thesis - Bachelor of Science in Engineering, Specialisation in Computer Engineering
Main field of study: Computer Engineering
Credits: 15 hp (ECTS)
Semester, Year: Spring, 2024
Supervisor: Emin Zerman
Examiner: Stefan Forsström
Course code: DT099G
Programme: Degree of Bachelor of Science with a major in Computer Engineering



Resources used:
https://opencv.org/
https://stackoverflow.com/
https://github.com
https://pieriantraining.com/kalman-filter-opencv-python-example/
https://docs.ultralytics.com/ Documentation YOLOv8
"""



from objectTrackerSetUp import ObjectTracker

if __name__ == "__main__":
    tracker = ObjectTracker('yolov8n.pt')
    tracker.run()
