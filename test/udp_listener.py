import socket

# Configuration: match the port you used in the iOS app
UDP_IP = "0.0.0.0"   # listen on all available network interfaces
UDP_PORT = 4399      # must match sender port

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for UDP packets on port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(8192)  # buffer size in bytes
    try:
        message = data.decode("utf-8")
    except UnicodeDecodeError:
        message = str(data)
    print(f"[{addr}] {message}")
