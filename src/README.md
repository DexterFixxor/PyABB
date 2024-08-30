# Structure


```python

class BaseJointInterface:
    """ 
    Sluzi za low level kontrolu zglobova prosledjenih 
    konstruktoru ukoliko se oni nalaze u parent-u.

    Funkcija reset_joint_state treba da inicijalizuje robota u "HOME" pozicije koje su postavljene u DH parametrima. TODO: ovo mozda prebaciti u JointInterface za postavljanje pocetnog polozaja

    Note: Trenutno je odradjeno samo zadavanje pozicije zglobovima
    """
    
class ROBOTNAME_Interface(BaseJointInterface):
    """
    Implementacija upravljanja ROBOTNAME instancom,
    zahteva da se definisu nazivi zglobova koji se prosledjuju super() klasi.

    URDF se ucitava van klase, pa se klient u kome je ucitano i ID koji mu je dodeljen prosledjuju Interface klasi.
    """

class Gripper_Interface(BaseJointInterface):
    """
        Gripper se sastoji od minimum 2 zgloba (levi, desni). 
        input args:
            gripper_config = {
                "parent_joint_name" :
                {
                    "mimic_1" : mimic_value,
                    "mimic_2" : mimic_value,
                }
            }

        __init__(self,...):
            mimic_scaling :np.array

        _apply_gripper_action(self, parent_angle):
            actions = parent_angle * self.mimic_scaling
            self.setJointTargetPosition(actions)

    """

class GRIPPERNAME_Interface(Gripper_interface):
    def __init__(self):
        """
        setup gripper config for specified gripper
        """
    def setGripperAction(self, action):
        """
        action should be clipped to range [0, 1]
        calculate angle for parent_joint_name,
        for example, rescale:
            angle = action * parent_joint_range + parent_joint_lower_limit
        """


class BaseTask:

    """
        Osnovna klasa za postavljanje scene:
        - ucitavanje osnovne scene: groundPlane
        - ucitavanje konfiguracionih parametara:
            * simulation_dt
            * rendering_dt (mora biti celobrojni umnozak simulation_dt)
            * state_clip
            * observation_clip
            * action_clip

        - abstraktne metode:
            - _get_observations() -> azurira internu vrednost observacija sa novim stanjima
            - _get_states() -> azurira internu vrednost stanja
            - _get_reward() -> racuna nagradu
            - _get_done() -> provera da li je doslo do kraja izvrsenja zadatka
    """


```

TODO:
* gripper - hvatanje kocke i pustanje kocke (sa ili bez attach funkcije)
* input u sim: delta_T, delta_Q end effector
* input u mrezu: 
    1. stanje zglobova, stanje grippera (1 ili 0), pozicija objekata u scenu, CILJ
    2. pozicija objekta je vizuelna, cilj vizuelni

* randomizacija domena:
    1. vizuali: teksture, boje robota, grippera, itd.


* eksperimenti na pravom robotu:
    1. kopmatibilnost inputa u sim i inputa u realnog robota
    2. Testiranje GOTO target bazirano na viziji
