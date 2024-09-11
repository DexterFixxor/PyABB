

class HERBuffer():

    def __init__(self, odnos = 0.5, batch_size = 256, buffer_len = 10e6):

        self.her_batch = int(odnos * batch_size)
        self.batch = int((1-odnos) * batch_size)

        # instanciras redovan buffer
        # instanciras HER buffer

        # cuvanje cele epizode


    def push(self, *transition):
        pass

    def reset_episode(self, reward_function):
        #calculate new rewards
        pass

    def sample(self):
        # sample redovan buffer
        # sample HER buffer
        #torch.cat() axis = 0
        #np.concatenate
        pass