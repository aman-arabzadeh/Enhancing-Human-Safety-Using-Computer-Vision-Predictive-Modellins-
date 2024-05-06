import tkinter as tk

# Initialize the main window
root = tk.Tk()
root.title("Apple Animation")

# Set the size of the window and the restricted area for movement
canvas_width = 1000
canvas_height = 700
restricted_width = 600
restricted_height = 400
start_x = (canvas_width - restricted_width) // 2  # Start position for x
start_y = (canvas_height - restricted_height) // 2  # Start position for y
end_x = start_x + restricted_width  # End position for x
end_y = start_y + restricted_height  # End position for y

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# Load the apple image (ensure the path is correct)
apple_image = tk.PhotoImage(file='apple.png')  # Update the path if needed
# Start the apple at the middle of the restricted area
apple = canvas.create_image((start_x + end_x) // 2, (start_y + end_y) // 2, image=apple_image)

# Animation settings
move_speed = 3
move_x = move_speed
gravity = 0.09  # Gravity effect
velocity_y = -3  #

def animate():
    global move_x, velocity_y
    # Get current position
    x, y = canvas.coords(apple)
    x_new = x + move_x
    # Calculate new vertical position using parabolic motion (projectile physics)
    velocity_y += gravity  # Gravity pulls the object down
    y_new = y + velocity_y


    if x_new >= end_x or x_new <= start_x:
        move_x = -move_x
    # Reverse direction if it hits the vertical boundaries
    if y_new >= end_y or y_new <= start_y:
        velocity_y = -velocity_y  # Invert velocity upon hitting a boundary

    # Move the apple
    canvas.move(apple, move_x, velocity_y)

    # Repeat the animation
    root.after(50, animate)  # Adjust the delay for faster or slower animation

# Start the animation
animate()

# Run the tkinter event loop
root.mainloop()
