import socket
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# UDP server configuration (same as the sending configuration in your existing script)
UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005     # The port number the server is sending data to

# Dictionary to store the positions history for each RX ID
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
            # Initialize list for new rx_id
            if rx_id not in positions:
                positions[rx_id] = []
            # Append the new position to the rx_id's position list
            positions[rx_id].append((x, y, z))
        else:
            print(f"Received invalid message: {message}")

def animate(i, ax):
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
    # Plot trajectories for each rx_id
    for rx_id, pos_list in positions.items():
        xs = [pos[0] for pos in pos_list]
        ys = [pos[1] for pos in pos_list]
        zs = [pos[2] for pos in pos_list]
        ax.plot(xs, ys, zs, label=f"RX {rx_id}")
        # Optionally, mark the current position with a scatter point
        ax.scatter(xs[-1], ys[-1], zs[-1], marker='o')
    ax.legend()

def main():
    # Start the UDP data receiving thread
    threading.Thread(target=receive_udp_data, daemon=True).start()

    # Set up the matplotlib figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Start the animation
    ani = animation.FuncAnimation(fig, animate, fargs=(ax,), interval=100)

    plt.show()

if __name__ == "__main__":
    main()