"""
API between your code and the duckiebot ros topics with Position PID control.
"""

import os
import time
import random
import socket
import curses
import numpy as np
import rospy
from enum import Enum
from duckietown_msgs.msg import WheelsCmdStamped
from duckietown_msgs.msg import WheelEncoderStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, Int32, Float32

class PIDController:
    """
    Implémentation personnalisée d'un contrôleur PID
    """
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=None, type=''):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint  # En mode position, c'est la position cible en ticks
        self.output_limits = output_limits
        self.type = type
        
        self.error_sum = 0
        self.last_error = 0
        self.last_time = rospy.Time.now()
    
    def compute(self, current_position):
        # Calculer le temps écoulé
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        if dt <= 0:
            return 0
            
        # Calculer l'erreur (différence entre position cible et position actuelle)
        error = self.setpoint - current_position
        
        # Terme proportionnel
        p_term = error
        
        # Terme intégral
        self.error_sum += error * dt
        i_term = self.error_sum
        
        # Terme dérivé (seulement si dt est significatif)
        d_term = 0
        if dt > 0:
            d_term = (error - self.last_error) / dt
        self.last_error = error
        
        # Calculer la sortie
        output = self.kp * p_term + self.ki * i_term + self.kd * d_term
        
        print('name:', self.type, 'position:', current_position, 'target:', self.setpoint, 'error:', error)
        print('p_term:', p_term, 'i_term:', i_term, 'd_term:', d_term, 'output:', output)

        # Appliquer les limites si définies
        if self.output_limits is not None:
            output = max(self.output_limits[0], min(output, self.output_limits[1]))
        
        return output
    
    def reset(self):
        """Réinitialiser l'intégrateur et l'erreur précédente"""
        self.error_sum = 0
        self.last_error = 0
    
    def update_gains(self, kp, ki, kd):
        """Mettre à jour les gains du PID"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
    def get_current_error(self):
        """Récupérer l'erreur actuelle pour la compensation"""
        return self.last_error

class DuckieBotAPI(object):
    """
    API between the code and the duckiebot ros topics.
    This class is an interface that defines usefull functions, used by the discrete actions and continuous
    actions environments. Enhanced with PID position control for wheel movement.
    """

    class Actions(Enum):
        FORWARD = 0
        BACKWARD = 1
        LEFT = 2
        RIGHT = 3
        STOP = 4  # Action pour arrêter le robot

    def __init__(self, **params):
        print()
        print("    ______________________________________________________    ")
        print()
        print("   ___                 _            _   _       ____  _     _ ")
        print("  |_ _|_ __         __| | ___ _ __ | |_| |__   |  _ \| |   | |")
        print("   | || '_ \ _____ / _` |/ _ \ '_ \| __| '_ \  | |_) | |   | |")
        print("   | || | | |_____| (_| |  __/ |_) | |_| | | | |  _ <| |___|_|")
        print("  |___|_| |_|      \__,_|\___| .__/ \__|_| |_| |_| \_\_____(_)")
        print("                             |_|                              ")
        print("    ______________________________________________________    ")
        print()
        print()
        self.robot_name = params.get("robot_name", "paperino")      # Duckiebot name
        self.fixed_linear_velocity: float = params.get("fixed_linear_velocity", 0.4)
        self.fixed_angular_velocity: float = params.get("fixed_angular_velocity", 0.2)
        self.control_time: float = params.get("control_time", 1)  # Temps de contrôle pour chaque action
        
        # Paramètres pour le déplacement en position
        self.position_increment = params.get("position_increment", 135)  # Nombre de ticks à avancer par action
        
        # Facteur de compensation pour les erreurs résiduelles
        self.error_compensation_factor = params.get("error_compensation_factor", 1.0)  # 1.0 = 100% de compensation

        # Init a node for this api
        print("  > Initializing node...")
        self.node = rospy.init_node('actions_converter', anonymous=True)
        print("  > Node initialized.")

        # PID control parameters
        self.TICKS_PER_REV = 135  # Nombre de ticks par tour de roue
        self.UPDATE_RATE = 25     # Hz
        
        # Définir les gains PID pour différents types de mouvement
        # Gains pour mouvements en ligne droite
        self.straight_kp = 0.005
        self.straight_ki = 0.0001
        self.straight_kd = 0.0005 #0.002
        
        # Gains pour rotations (peuvent nécessiter des valeurs différentes pour un meilleur contrôle)
        self.rotation_kp = 0.01    # Plus élevé pour une meilleure précision dans les virages
        self.rotation_ki = 0.0001  # Similaire ou légèrement plus faible
        self.rotation_kd = 0.001   # Plus élevé pour limiter les dépassements
        
        # Initialiser avec les gains par défaut (ligne droite)
        self.pid_left = PIDController(
            kp=self.straight_kp, 
            ki=self.straight_ki, 
            kd=self.straight_kd, 
            setpoint=0,  # Position cible initiale
            output_limits=(-0.75, 0.75),
            type='left'
        )
        self.pid_right = PIDController(
            kp=self.straight_kp, 
            ki=self.straight_ki, 
            kd=self.straight_kd, 
            setpoint=0,  # Position cible initiale
            output_limits=(-0.75, 0.75),
            type='right'
        )
        
        # Position actuelle des roues en ticks
        self.current_ticks_left = 0
        self.current_ticks_right = 0
        
        # Position cible en ticks
        self.target_position_left = 0
        self.target_position_right = 0
        
        # Erreur résiduelle à compensée
        self.residual_error_left = 0
        self.residual_error_right = 0
        
        # État du contrôle PID
        self.pid_active = True  # Par défaut, le PID est actif
        
        # Indicateur d'action complétée
        self.action_completed = False
        
        # Gestion du timeout directement dans pid_control
        self.last_action_time = rospy.Time.now()
        self.action_active = False
        
        # Type de mouvement actuel (pour les gains PID)
        self.current_movement_type = "straight"  # "straight" ou "rotation"
        
        # Actions possibles - Mappées à des incréments de position
        self.ACTION_MAPPING = {
            0: (0.75, 0.75),         # avancer: les deux roues avancent d'une même valeur
            1: (-0.75, -0.75),       # reculer: les deux roues reculent d'une même valeur
            2: (0.175, -0.175),  # droite: roue gauche avance, roue droite recule
            3: (-0.175, 0.175),  # gauche: roue gauche recule, roue droite avance
            4: (0, 0)          # stop: pas de changement de position cible
        }
        
        # Définir les types de mouvement pour chaque action
        self.MOVEMENT_TYPES = {
            0: "straight",  # avancer: mouvement en ligne droite
            1: "straight",  # reculer: mouvement en ligne droite
            2: "rotation",  # droite: rotation
            3: "rotation",  # gauche: rotation
            4: "stop"       # stop: pas de mouvement
        }
        
        # Temps pour le calcul
        self.last_time = rospy.Time.now()
        
        # Setup ros command publisher
        self.commands_publisher = rospy.Publisher('/' + str(self.robot_name) + '/wheels_driver_node/wheels_cmd',
                                                  WheelsCmdStamped, queue_size=10)
        print("  > Commands publisher initialized.")

        # Setup ros command publisher
        self.observations_publisher = rospy.Publisher('/' + str(self.robot_name) + '/observation', CompressedImage, queue_size=10)
        print("  > Observation publisher initialized.")
        self.last_observation_message = None

        # Set up the observation update process
        self.last_observation_message = None
        self.observation_subscriber = rospy.Subscriber(
            f"/{self.robot_name}/camera_node/image/compressed",
            CompressedImage,
            self.observation_callback
        )

        # Setup action listener
        self.actions_subscriber = rospy.Subscriber('/' + str(self.robot_name) + '/discrete_action', Int32, self.actions_callback)
        
        # Setup wheel tick subscribers for PID control
        rospy.Subscriber('/' + str(self.robot_name) + '/left_wheel_encoder_node/tick', WheelEncoderStamped, self.left_tick_callback)
        rospy.Subscriber('/' + str(self.robot_name) + '/right_wheel_encoder_node/tick', WheelEncoderStamped, self.right_tick_callback)
        
        # Initialiser la position actuelle comme position cible initiale
        self.target_position_left = self.current_ticks_left
        self.target_position_right = self.current_ticks_right
        self.pid_left.setpoint = self.target_position_left
        self.pid_right.setpoint = self.target_position_right
        
        # stop robot first 
        self.stop_robot_direct()
        
        # Timer for PID control
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.UPDATE_RATE), self.pid_control)

        time.sleep(0.5)  # Wait for the publisher and subscriber to be registered.
        print("  > Api initialized with custom PID position control and adaptive gains.")
        rospy.spin()

    def observation_callback(self, observation_message):
        """
        This function is called everytime an observation is received.
        Returns: None
        """
        try:
            self.last_observation_message = observation_message
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def left_tick_callback(self, msg):
        """Callback pour les ticks de la roue gauche"""
        self.current_ticks_left = msg.data
    
    def right_tick_callback(self, msg):
        """Callback pour les ticks de la roue droite"""
        self.current_ticks_right = msg.data
    
    def update_pid_gains(self, movement_type):
        """Mettre à jour les gains PID en fonction du type de mouvement"""
        if movement_type == "straight":
            self.pid_left.update_gains(self.straight_kp, self.straight_ki, self.straight_kd)
            self.pid_right.update_gains(self.straight_kp, self.straight_ki, self.straight_kd)
            # self.pid_left.output_limits = (-0.25, 0.25)
            # self.pid_right.output_limits = (-0.25, 0.25)
            
            rospy.loginfo("PID gains set for straight movement")
        elif movement_type == "rotation":
            self.pid_left.update_gains(self.rotation_kp, self.rotation_ki, self.rotation_kd)
            self.pid_right.update_gains(self.rotation_kp, self.rotation_ki, self.rotation_kd)
            rospy.loginfo("PID gains set for rotation movement")
    
    def store_residual_errors(self):
        """
        Stocker les erreurs résiduelles actuelles pour la compensation 
        dans les prochaines actions
        """
        if self.action_active:
            # Calculer l'erreur résiduelle entre la position cible et la position actuelle
            self.residual_error_left = self.target_position_left - self.current_ticks_left
            self.residual_error_right = self.target_position_right - self.current_ticks_right
            
            rospy.loginfo(f"Stored residual errors - Left: {self.residual_error_left}, Right: {self.residual_error_right}")
    
    def stop_robot_direct(self):
        """Arrêter le robot directement"""
        # Stocker les erreurs résiduelles avant d'arrêter
        self.store_residual_errors()
        
        # Désactiver le PID temporairement
        self.pid_active = False
        
        # Envoyer directement la commande d'arrêt
        self.set_velocity_raw(0.0, 0.0)
        
        # Réinitialiser les intégrateurs PID
        self.pid_left.reset()
        self.pid_right.reset()
        
        # Définir la position actuelle comme la position cible
        # Cela évite que le robot essaie de retourner à une position précédente
        self.target_position_left = self.current_ticks_left
        self.target_position_right = self.current_ticks_right
        self.pid_left.setpoint = self.target_position_left
        self.pid_right.setpoint = self.target_position_right
        
        self.action_active = False
        self.action_completed = True
        rospy.loginfo("Robot stopped directly and position target reset")
        
        # Ne pas réactiver le PID tout de suite
        # self.pid_active = True
    
    def publish_observation(self):
        """Publier l'observation actuelle"""
        if self.last_observation_message:
            if not rospy.is_shutdown():
                self.observations_publisher.publish(self.last_observation_message)
                rospy.loginfo("Observation published")
    
    def actions_callback(self, data):
        """Callback pour traiter les actions reçues"""
        action = int(data.data)
        print("    [api] Received action", action)
        if isinstance(action, np.ndarray):
            action = int(action)
        
        # Appliquer l'action
        self.apply_action(action)
        
    def apply_action(self, action):
        """Appliquer une action spécifique en définissant une nouvelle position cible"""
        # Traiter la nouvelle action
        if action in self.ACTION_MAPPING:
            # Obtenir les facteurs de direction pour chaque roue
            left_factor, right_factor = self.ACTION_MAPPING[action]
            
            if action == 4:  # STOP
                self.stop_robot_direct()
                return
            
            # Mettre à jour le type de mouvement et les gains PID
            movement_type = self.MOVEMENT_TYPES.get(action, "straight")
            self.current_movement_type = movement_type
            self.update_pid_gains(movement_type)
            
            # Calculer les nouvelles positions cibles avec compensation d'erreur
            base_increment_left = left_factor * self.position_increment
            base_increment_right = right_factor * self.position_increment
            
            # Ajouter la compensation d'erreur aux incréments de base
            # Si le facteur est négatif, l'erreur doit aussi être inversée
            if left_factor != 0:  # Éviter la division par zéro
                error_adjustment_left = self.residual_error_left * self.error_compensation_factor * (left_factor/abs(left_factor))
            else:
                error_adjustment_left = 0
                
            if right_factor != 0:  # Éviter la division par zéro
                error_adjustment_right = self.residual_error_right * self.error_compensation_factor * (right_factor/abs(right_factor))
            else:
                error_adjustment_right = 0
            
            # Appliquer les ajustements
            total_increment_left = base_increment_left + error_adjustment_left
            total_increment_right = base_increment_right + error_adjustment_right
            
            rospy.loginfo(f"Base increments - Left: {base_increment_left}, Right: {base_increment_right}")
            rospy.loginfo(f"Error adjustments - Left: {error_adjustment_left}, Right: {error_adjustment_right}")
            rospy.loginfo(f"Total increments - Left: {total_increment_left}, Right: {total_increment_right}")
            
            # Calculer les positions cibles finales
            self.target_position_left = self.current_ticks_left + total_increment_left
            self.target_position_right = self.current_ticks_right + total_increment_right
            
            # Mettre à jour les setpoints des PIDs
            self.pid_left.setpoint = self.target_position_left
            self.pid_right.setpoint = self.target_position_right
            
            # Réinitialiser les intégrateurs pour éviter l'accumulation d'erreur
            self.pid_left.reset()
            self.pid_right.reset()
            
            # Réactiver le PID si nécessaire
            self.pid_active = True
            self.action_completed = False
            
            # Réinitialiser les erreurs résiduelles après les avoir utilisées
            self.residual_error_left = 0
            self.residual_error_right = 0
            
            # Enregistrer le temps de début de l'action et marquer comme active
            self.last_action_time = rospy.Time.now()
            self.action_active = True
            
            rospy.loginfo(f"New target positions - Left: {self.target_position_left}, Right: {self.target_position_right} (Movement type: {movement_type})")
        else:
            rospy.logwarn(f"Unknown action received: {action}")
            self.stop_robot_direct()

    def is_position_reached(self):
        """Vérifie si la position cible a été atteinte avec une certaine tolérance"""
        tolerance = 5  # Tolérance en ticks
        left_error = abs(self.target_position_left - self.current_ticks_left)
        right_error = abs(self.target_position_right - self.current_ticks_right)
        
        return left_error < tolerance and right_error < tolerance

    def pid_control(self, event):
        """Fonction principale d'asservissement PID en position"""
        current_time = rospy.Time.now()
        
        # Vérification du timeout de l'action et de l'atteinte de la position
        if self.action_active:
            # Vérifier si la position cible est atteinte
            if self.is_position_reached() and not self.action_completed:
                print('POSITION TARGET REACHED ###################################################')
                self.action_completed = True
                self.stop_robot_direct()  # Cela stockera aussi les erreurs résiduelles
                self.publish_observation()
                return
            
            # Vérifier le timeout
            elapsed_time = (current_time - self.last_action_time).to_sec()
            if elapsed_time >= self.control_time and not self.action_completed:
                print('ACTION TIMEOUT ###################################################')
                self.stop_robot_direct()  # Cela stockera aussi les erreurs résiduelles
                self.publish_observation()
                return
        
        # Ne pas calculer le PID si le PID est désactivé ou si l'action est complétée
        if not self.pid_active or self.action_completed:
            return
        
        # Calculer les commandes PID basées sur la position actuelle
        cmd_left = self.pid_left.compute(self.current_ticks_left)
        cmd_right = self.pid_right.compute(self.current_ticks_right)
        
        # Appliquer les commandes aux roues
        self.set_velocity_raw(cmd_left, cmd_right)
            
        rospy.logdebug(f"Position - Left: {self.current_ticks_left}, Right: {self.current_ticks_right}")
        rospy.logdebug(f"Command - Left: {cmd_left}, Right: {cmd_right}")

    def set_velocity_raw(self, left_wheel_velocity=0.0, right_wheel_velocity=0.0):
        # print("    [api] setting vel raw to ", left_wheel_velocity, ", ", right_wheel_velocity) 
        msg = WheelsCmdStamped()

        # Set message parameters
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.vel_left = left_wheel_velocity
        msg.vel_right = right_wheel_velocity

        # Publish the message
        if not rospy.is_shutdown():
            self.commands_publisher.publish(msg)
            
    def set_velocity(self, linear_velocity=0.0, angular_velocity=0.0):
        """
        Cette méthode n'est plus utilisée avec l'asservissement en position,
        mais conservée pour compatibilité avec d'autres codes éventuels.
        """
        rospy.logwarn("set_velocity is called, but the robot is using position control, not velocity control")
        # Rien à faire, car nous utilisons maintenant l'asservissement en position

if __name__ == "__main__":
    DuckieBotAPI(robot_name="paperino")
