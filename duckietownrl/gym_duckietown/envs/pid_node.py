import rospy
import numpy as np
import rospy
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown_msgs.msg import Twist2DStamped

# Définition des gains PID
KP = 1.0  # gain proportionnel
KI = 0.1  # gain intégral
KD = 0.01  # gain dérivé

# Définition des paramètres de la roue
INCREMENT_PAR_TOUR = 135.0  # incrément par tour de roue
FREQUENCE_PID = 30.0  # fréquence de mise à jour du PID (Hz)

class NoeudPID:
    def __init__(self):
        self.left_wheel_tick = 0
        self.right_wheel_tick = 0
        self.cmd_agent = np.array([0.0, 0.0])  # vitesse angulaire demandée pour chaque roue
        self.error_left = 0.0
        self.error_right = 0.0
        self.integrale_left = 0.0
        self.integrale_right = 0.0
        self.derniere_vitesse_left = 0.0
        self.derniere_vitesse_right = 0.0

        # Abonnement aux topics
        self.sub_left_wheel_tick = rospy.Subscriber('/paperino/left_wheel_encoder_node/tick', Twist2DStamped, self.callback_left_wheel_tick)
        self.sub_right_wheel_tick = rospy.Subscriber('/paperino/right_wheel_encoder_node/tick', Twist2DStamped, self.callback_right_wheel_tick)
        self.sub_cmd_agent = rospy.Subscriber('cmd_agent', WheelsCmdStamped, self.callback_cmd_agent)

        # Publication des commandes de vitesse
        self.pub_cmd_moteur = rospy.Publisher('/paperino/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=10)

    def callback_left_wheel_tick(self, msg):
        self.left_wheel_tick = msg.data

    def callback_right_wheel_tick(self, msg):
        self.right_wheel_tick = msg.data

    def callback_cmd_agent(self, msg):
        self.cmd_agent = np.array([msg.x, msg.y])

    def pid_update(self):
        # Calcul de la vitesse réelle de chaque roue
        vitesse_left = (self.left_wheel_tick - self.derniere_vitesse_left) * INCREMENT_PAR_TOUR / FREQUENCE_PID
        vitesse_right = (self.right_wheel_tick - self.derniere_vitesse_right) * INCREMENT_PAR_TOUR / FREQUENCE_PID

        # Calcul de l'erreur pour chaque roue
        self.error_left = self.cmd_agent[0] - vitesse_left
        self.error_right = self.cmd_agent[1] - vitesse_right

        # Calcul de l'intégrale pour chaque roue
        self.integrale_left += self.error_left * (1.0 / FREQUENCE_PID)
        self.integrale_right += self.error_right * (1.0 / FREQUENCE_PID)

        # Calcul de la dérivée pour chaque roue
        derivée_left = (self.error_left - self.derniere_vitesse_left) * FREQUENCE_PID
        derivée_right = (self.error_right - self.derniere_vitesse_right) * FREQUENCE_PID

        # Calcul de la commande de vitesse pour chaque roue
        cmd_left = KP * self.error_left + KI * self.integrale_left + KD * derivée_left
        cmd_right = KP * self.error_right + KI * self.integrale_right + KD * derivée_right

        # Publication de la commande de vitesse
        self.pub_cmd_moteur.publish(WheelsCmdStamped(header=None, vel_left=cmd_left, vel_right=cmd_right))

        # Mise à jour des valeurs précédentes
        self.derniere_vitesse_left = vitesse_left
        self.derniere_vitesse_right = vitesse_right

if __name__ == '__main__':
    rospy.init_node('noeud_pid')
    rate = rospy.Rate(FREQUENCE_PID)
    noeud = NoeudPID()

    while not rospy.is_shutdown():
        noeud.pid_update()
        rate.sleep()
