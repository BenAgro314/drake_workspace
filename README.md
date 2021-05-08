# drake_workspace

## Setting up and using Docker container



Navigate to the `docker_scripts` directory and run `./docker_build.sh your_password`. Then run `./docker_run.sh`.

To use a GUI in the docker container, on your local device add this to `~/.ssh/config`, replacing your information as necessary:

```
Host computername
    Hostname myip # put your host ip
    User username # put your username
    Port myport  # put whatever your port is 

Host drake-workspace
    ProxyCommand ssh -q computername nc -q0 localhost 2400 $ this is the port that is forwarded from computername to the docker container"
    LocalForward 5901 localhost:5901 # the port used by tigervncserver
    User username # put your username
```

Then, on your local device run `ssh drake-workspace` and enter the password you used when building the docker container.Then using a vnc viewer, connect to localhost:5901. Now you should be able to use the docker container as a remote desktop.




