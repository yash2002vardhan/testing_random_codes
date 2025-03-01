import subprocess

def get_wifi_passwords():
    try:
        result = subprocess.run(['netsh', 'wlan show profiles'], capture_output=True, text=True)
        profiles = result.stdout.split('\n')
        passwords = []

        for profile in profiles:
            if 'All User Profile' in profile:
                ssid = profile.split(':')[1].strip()
                try:
                    result = subprocess.run(['netsh', 'wlan show profile "' + ssid + '" key=clear'], capture_output=True, text=True)
                    password = result.stdout.split('Key Content')[1].split(':')[1].strip()
                    passwords.append((ssid, password))
                except Exception as e:
                    print(f"Error getting password for {ssid}: {e}")

        return passwords

    except Exception as e:
        print(f"An error occurred: {e}")

wifi_passwords = get_wifi_passwords()
for ssid, password in wifi_passwords: #type: ignore
    print(f"SSID: {ssid}, Password: {password}")
