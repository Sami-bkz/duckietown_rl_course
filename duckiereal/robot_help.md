# HELP!

Password of both robot is `quackquack`. The login is `duckie`. If your password is refused several time make sure you 
typed `ssh duckie@$ROBOT_NAME.local` and not `ssh $ROBOT_NAME.local` (it even happens to the best of us ...).

## Can't ping the robot

Make sure that you are on the same network than the robot.
If `ping $ROBOT_NAME.local` still doesn't work, it might be an ip issue: 
 - Make sure you DON'T have $ROBOT_NAME.local in /etc/hosts, set to a specific ip. The robot ip might change everytime 
it reconnect to the router.
 - Make sure avahi-daemon is running. It is necessary to retrieve the ip associated to $ROBOT_NAME.local from the LAN.
   - Verify if installed: `dpkg -l | grep avahi-daemon`
   - install it (avahi require dbus service): `sudo apt install -y dbus avahi-daemon`
   - launch both services `service dbus start && service avahi-daemon start`
If it still doesn't work ... call me.

## I can ping the robot but can't see topics

If `rostopic list` don't show topics (or like 2 or 3), it is probably an environment variables issues.
Type `hostname -I` in you container.

Verify the environment variables values:
```shell
echo "ROBOT_NAME = $ROBOT_NAME"
echo "ROS_MASTER_URI = $ROS_MASTER_URI"
echo "ROS_HOSTNAME = $ROS_HOSTNAME"
echo "ROS_IP = $ROS_IP"
```
Makes sure `$ROBOT_NAME` have for value the name of the robot ("paperino" or "gastone")
`$ROS_MASTER_URI` should have for value `http://$ROBOT_NAME.local:11311/`. 
`$ROS_HOSTNAME` and `$ROS_IP` should have the same IP that the first one shown by `hostname -I`. DO NOT set them to 
127.0.0.1. This ip can be used by your pc but for the robot, it refers to himself so it doesn't work.

## I can see topics but the robot doesn't execute my actions

`rostopic echo /$ROBOT_NAME/discrete_actions`, and then, in another terminal, try to publish actions on this topic 
using `rostopic pub --once std_msgs/Int32 "data: 0"`.

If it works, the api and the network and the connexion work, makes sure your script publish on the right topic.
If it doesn't, try to show the velocity values send to the robot by the api:
`rostopic echo /$ROBOT_NAME/wheels_driver_node/wheels_cmd`. You can also see it in the dashboard of the robot (in the 
webpage, type paperino.local in your web browser).

If there is no messages, ssh the robot, exec the container "duckiebot_interface" and type "ps aux | grep api".
If it shows more than one program, call me, if there is only one grep command, type "python3 /api.py" in a terminal and 
let it run while you do your stuff in another one.

If everything is fine but you still have the issue, do the steps in the previous section, if it still doesn't work, call 
me.
