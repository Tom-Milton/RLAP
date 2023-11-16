from .base import Controller
from .base import Action
import logging

logger = logging.getLogger(__name__)


class PIDController(Controller):
    def __init__(self, P=1, I=0, D=0, target=112):
        self.P = P
        self.I = I
        self.D = D
        self.target = target
        self.integrated_state = 0
        self.prev_state = -1

    def select_action(self, observation, **kwargs):
        # BG is the only state for this PID controller
        bg, sample_time = observation

        # Corrected previous state at start of operation (previously 0 not -1)
        # This way we update the prev_state to be the current state if at start of simulation
        # Otherwise a prev_state of 0 implies an initial very low glucose
        # When the first actual state comes in the controller thinks it has seen a large bg increase
        if self.prev_state == -1: self.prev_state = bg

        control_input = self.P * (bg - self.target) + \
            self.I * self.integrated_state + \
            self.D * (bg - self.prev_state) / sample_time

        logger.info('Control input: {}'.format(control_input))

        # update the states
        self.prev_state = bg
        self.integrated_state += (bg - self.target) * sample_time
        logger.info('prev state: {}'.format(self.prev_state))
        logger.info('integrated state: {}'.format(self.integrated_state))

        # return the action
        action = control_input
        return action

    def reset(self):
        self.integrated_state = 0
        self.prev_state = -1
