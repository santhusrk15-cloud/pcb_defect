from ftplib import FTP

# Function to connect and perform FTP operations
def connect_and_execute_ftp(server, operations):
    try:
        with FTP(server) as ftp:
            ftp.login()  # Attempt anonymous login
            print(f"Connected to {server}")
            
            for operation in operations:
                operation(ftp)
            
    except Exception as e:
        print(f"Error connecting to {server}: {e}")

# Operation functions for specific FTP tasks
def list_dir(ftp):
    print("Listing directory contents:")
    ftp.dir()

def change_and_list_dir(ftp, directory):
    try:
        ftp.cwd(directory)
        print(f"Changed directory to {directory}")
        ftp.dir()
    except Exception as e:
        print(f"Could not change directory: {e}")

def upload_file(ftp, remote_name, local_path):
    with open(local_path, 'rb') as file:
        ftp.storlines(f'STOR {remote_name}', file)
        print(f"Uploaded file as {remote_name}")

def download_file(ftp, remote_name, local_path):
    with open(local_path, 'wb') as file:
        ftp.retrbinary(f'RETR {remote_name}', file.write)
        print(f"Downloaded file to {local_path}")

# Example FTP connections and operations
connect_and_execute_ftp('ftp.cs.brown.edu', [
    list_dir,
    lambda ftp: change_and_list_dir(ftp, '/incoming')
])

connect_and_execute_ftp('ftp1.at.proftpd.org', [
    list_dir,
])

connect_and_execute_ftp('ftp.ed.ac.uk', [
    list_dir,
    lambda ftp: change_and_list_dir(ftp, 'pub'),
    lambda ftp: download_file(ftp, 'convsa_62_guide.pdf', 'c.pdf')
])
