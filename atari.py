import gym
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.optimizers import Adam
from keras.models import Model,Sequential,load_model
import cv2
import random
import numpy as np
import keras
import tensorflow as tf

memory = []
memory_length = 1000000
no_training_frames = 30000000
training_start = 50000
learning_rate = 0.00001
env = gym.make('BreakoutDeterministic-v4')
num_of_actions = env.action_space.n
epsilon = 1
epsilon_final = 0.1
render =  False
exploration_steps =200000
epsilon_decay = (epsilon - epsilon_final)/exploration_steps
batch_size = 32
ATARI_SHAPE =[84,84,4]
alpha = 0.1
gamma = 0.99
TAU = 0.01

def create_model():

	model = Sequential()
	model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
	                 input_shape=ATARI_SHAPE))
	model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(num_of_actions))
	model.summary()
	optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(optimizer, loss=tf.keras.losses.Huber(delta=100.0))
	return model


model = create_model()
target_model = create_model()

def get_action(present_state = None):
		p = random.random()
		if(len(memory)< training_start):
			#print("explore")
			return random.randrange(num_of_actions)
		#### explore part ####
		if(p < epsilon):
			action = random.randrange(num_of_actions)
			#print("explore")
		#### greedy part ####
		else:
			action = np.argmax(model.predict(np.expand_dims(present_state,axis=0)))
			#print("greedy")
		return action




def target_train():
	model_weights = model.get_weights()
    target_model_weights = target_model.get_weights()
    for i in range(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
    target_model.set_weights(target_model_weights)


def train():
	if(len(memory) < training_start):
		return 0
	
	global epsilon,epsilon_final,epsilon_final
	
	if(epsilon > epsilon_final):
		epsilon = epsilon - epsilon_decay
	
	mini_batch = random.sample(memory, batch_size)
	S_t_copy = np.zeros((batch_size,ATARI_SHAPE[0],ATARI_SHAPE[1],ATARI_SHAPE[2]))
	S_t_1_copy = np.zeros((batch_size,ATARI_SHAPE[0],ATARI_SHAPE[1],ATARI_SHAPE[2]))
	reward_copy = np.zeros((batch_size))
	A_t = np.zeros((batch_size,num_of_actions),np.int32)
	done_copy = []
	target = np.zeros((batch_size,num_of_actions))
	y = np.zeros((batch_size))

	
	for i in range(batch_size):
		S_t_copy[i] =	 mini_batch[i][0]
		reward_copy[i] = mini_batch[i][1]
		S_t_1_copy[i] = mini_batch[i][2]
		A_t[i] = mini_batch[i][3]
		done_copy.append(mini_batch[i][4])
	#print(np.array_equal(S_t_1_copy[0],S_t_1_copy[1]))

	Q_t_1 = target_model.predict(S_t_1_copy)
	#print(Q_t_1)
	#print(Q_t_1[0],Q_t_1[1])
	for i in range(batch_size):
		if(done_copy[i] == True):
			target[i][A_t[i]] = reward_copy[i]
		else:
			target[i][A_t[i]] = reward_copy[i] + gamma * np.max(Q_t_1[i])
	#print(target)
	h = model.fit(S_t_copy, target, epochs=1,batch_size=batch_size, verbose=0)
	
	return float(h.history['loss'][0])


def preprocessing(image):
	image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	image = image[34:, 0:]
	image = cv2.resize(image,(84,84))
	return image

def main():

	
	#model = create_model()
 
 
	i = 0
	episode = 0
	while(i < no_training_frames):
		episode_reward = 0
		observation = env.reset()
		observation = preprocessing(observation)
		S_t = np.stack((observation,observation,observation,observation),axis= 2)
		done = False
		loss = [0.0]
		while(done == False):
			if(render):
				env.render()
			action = get_action(S_t)
			observation, reward, done, __ = env.step(action)
			i = i + 1
			observation = preprocessing(observation)
			S_t_1  = np.stack((S_t[:,:,0],S_t[:,:,1],S_t[:,:,2],observation),axis= 2)
			if(len(memory) == memory_length):
				memory.pop(0)
			memory.append((S_t,reward,S_t_1,action,done))
			if(i % 4 == 0):
				loss.append(train())
				target_train()
			episode_reward += reward
			S_t = S_t_1
		episode = episode + 1
		avg_loss = np.mean(np.asarray(loss,np.float32))
		print("episode no:",episode,"Frame No:",i,"episode_reward:",episode_reward,"epsilon:",epsilon,"memory length:",len(memory),"avg loss:",avg_loss)
		if(i % 200 ==0 and i > 100):
			model.save('atari_'+str(i))

main()





















