import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# UDP server configuration (same as the sending configuration in your existing script)
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005     # The port number the server is sending data to

# Dictionary to store the latest positions for each RX ID
positions = {}

def receive_udp_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8').strip()
        # Expected message format: 'rx_id,x,y,z,speed'
        parts = message.split(',')
        if len(parts) == 5:
            rx_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            speed = float(parts[4])
            positions[rx_id] = (x, y, z)
        else:
            print(f"Received invalid message: {message}")

def animate(i, scatter, ax):
    # Clear the axes
    ax.clear()
    # Set the axes limits (adjust as needed)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2)
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Plot each RX's current position
    for rx_id, pos in positions.items():
        x, y, z = pos
        ax.scatter(x, y, z, label=f"RX {rx_id}")
    ax.legend()

def main():
    # Start the UDP data receiving thread
    threading.Thread(target=receive_udp_data, daemon=True).start()

    # Set up the matplotlib figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Dummy initial scatter plot
    scatter = ax.scatter([], [], [])

    # Start the animation
    ani = animation.FuncAnimation(fig, animate, fargs=(scatter, ax), interval=100)

    plt.show()

if __name__ == "__main__":
    main()