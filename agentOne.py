import numpy as np
import datetime
import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

# 환경 및 학습 하이퍼파라머 설정 
state_size = 82  # RayPerceptionSensor의 크기로 설정
continuous_action_size = 2

load_model = False
train_mode = True

# 수정된 하이퍼파라머터
discount_factor = 0.99
learning_rate = 4e-5  # 낮은 학습률로 안정화
n_step = 64  # 더 길 N-Step Return 사용
batch_size = 256  # 더 큰 배치 크기 사용
n_epoch = 10  # 에포크 수 증가
epsilon = 0.1  # 클리피버의 범위 줄이기
entropy_bonus = 0.001  # 엔트로피 보너스 낮게 설정
critic_loss_weight = 0.1  # Critic 손실에 과중치 증가

# 그래디열트 클링필 및 학습 스텔 설정
grad_clip_max_norm = 0.1
run_step = 2000000 if train_mode else 0
test_step = 100000

print_interval = 100
save_interval = 1000

# 모듈 저장 경로 및 장치 설정
env_name = "IDIOTmlagent"
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = os.path.join(".", "saved_models", env_name, "PPO", date_time)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ActorCritic 네트워크 정의
class ActorCritic(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, 256)
        self.bn1 = torch.nn.LayerNorm(256)
        self.d2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.LayerNorm(256)
        self.pi = torch.nn.Linear(256, continuous_action_size)
        self.v = torch.nn.Linear(256, 1)

        torch.nn.init.kaiming_uniform_(self.d1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.d2.weight, nonlinearity='relu')
        torch.nn.init.xavier_uniform_(self.pi.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.d1(x)))
        x = F.relu(self.bn2(self.d2(x)))
        return torch.tanh(self.pi(x)), self.v(x)

# PPOAgent 클래스 정의
class PPOAgent:
    def __init__(self):
        self.network = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=5000, gamma=0.9)
        self.memory = []
        self.writer = SummaryWriter(save_path)

    # 에이전트의 활동 결정
    def get_action(self, state, training=True):
        self.network.train(training)
        with torch.no_grad():
            continuous_action, _ = self.network(torch.FloatTensor(state).to(device))
            continuous_action = continuous_action.cpu().numpy()
        return continuous_action

    # 메모리 업데이트 및 학습 함수
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        if len(self.memory) < batch_size:
            return None, None

        state_shape = self.memory[0][0].shape
        continuous_action_shape = self.memory[0][1].shape if isinstance(self.memory[0][1], np.ndarray) else (1,)

        state = np.stack([np.resize(m[0], state_shape) for m in self.memory])
        continuous_action = np.stack([np.resize(m[1], continuous_action_shape) for m in self.memory])
        reward = np.stack([m[2] for m in self.memory])
        next_state = np.stack([np.resize(m[3], state_shape) for m in self.memory])
        done = np.stack([m[4] for m in self.memory])
        self.memory.clear()

        state, continuous_action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                    [state, continuous_action, reward, next_state, done])

        _, value = self.network(state)
        _, next_value = self.network(next_state)

        # value와 next_value의 차원을 맞쿀기 위해 squeeze 사용
        value = value.squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        next_value = next_value.squeeze(-1)  # (batch_size, 1) -> (batch_size,)

        # Advantage와 Target 계산
        delta = reward + (1 - done) * discount_factor * next_value - value
        advantage = delta.clone()
        target = advantage + value

        # Target 및 Advantage 클립픽 적용
        target = torch.clamp(target, -10, 10)
        advantage = torch.clamp(advantage, -10, 10)

        # PPO Actor-Critic 손실 계산
        old_log_probs = -0.5 * ((continuous_action - value.unsqueeze(-1)) ** 2).sum(dim=-1).detach()
        new_log_probs = -0.5 * ((continuous_action - value.unsqueeze(-1)) ** 2).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        actor_loss = -torch.min(ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage).mean()

        critic_loss = F.mse_loss(value, target).mean()

        entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()
        total_loss = actor_loss + critic_loss_weight * critic_loss - entropy_bonus * entropy

        # 그래디열트 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=grad_clip_max_norm)
        self.optimizer.step()
        self.scheduler.step()

        return actor_loss.item(), critic_loss.item()

    # 모듈 저장 및 로그 기록
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        # Save as PyTorch model
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

        # Save as ONNX model
        dummy_input = torch.randn(1, state_size).to(device)
        self.save_onnx_model(self.network, dummy_input, save_path + "/model.onnx")

    def save_onnx_model(self, model, dummy_input, file_path):
        model.eval()  # 평가 모드로 설정
        try:
            torch.onnx.export(model,
                              dummy_input,
                              file_path,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

            # ONNX 파일에 추가적인 메타데이터 설정
            import onnx
            onnx_model = onnx.load(file_path)
            meta_keys = [prop.key for prop in onnx_model.metadata_props]
            if "version_number" not in meta_keys:
                meta = onnx.ModelProto.MetadataPropsEntry()
                meta.key = "version_number"
                meta.value = "1.0"
                onnx_model.metadata_props.append(meta)
            onnx.save(onnx_model, file_path)

            print(f"ONNX model saved at {file_path}")
        except Exception as e:
            print(f"Failed to export ONNX model: {e}")

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        if actor_loss is not None and critic_loss is not None:
            self.writer.add_scalar("model/actor_loss", actor_loss, step)
            self.writer.add_scalar("model/critic_loss", critic_loss, step)

if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    environment_parameters_channel = EnvironmentParametersChannel()
    env = UnityEnvironment(file_name=env_name,
                           worker_id=1,
                           side_channels=[engine_configuration_channel,
                                          environment_parameters_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    print("Observation Specs:")
    dec, term = env.get_steps(behavior_name)
    for idx, obs_spec in enumerate(spec.observation_specs):
        print(f"Observation {idx}: Shape = {obs_spec.shape}, Type = {obs_spec.observation_type}")

    engine_configuration_channel.set_configuration_parameters(time_scale=20.0)

    agent = PPOAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=20.0)

        dec, term = env.get_steps(behavior_name)

        if len(dec) == 0 and len(term) > 0:
            state = term.obs[0]
        elif len(dec) > 0:
            state = dec.obs[0]
        else:
            continue

        state = np.clip(state, -1, 1)
        continuous_action = agent.get_action(state, train_mode)

        # continuous_action이 비어 있거나 크기가 다르다면 기본값으로 초기화
        if continuous_action is None or len(continuous_action.shape) == 0:
            continuous_action = np.zeros(continuous_action_size)

        continuous_action = np.expand_dims(continuous_action, axis=0) if len(continuous_action.shape) == 1 else continuous_action
        action_tuple = ActionTuple()
        action_tuple.add_continuous(continuous_action)

        if len(dec) > 0:
            env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term) > 0
        next_state = term.obs[0] if done else dec.obs[0]
        next_state = np.clip(next_state, -1, 1)
        reward = term.reward[0] if done else dec.reward[0]
        score += reward

        if train_mode:
            agent.append_sample(state, continuous_action, [reward], next_state, [done])
            if (step + 1) % n_step == 0:
                result = agent.train_model()
                if result is not None:
                    actor_loss, critic_loss = result
                    if actor_loss is not None and critic_loss is not None:
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
        if done:
            episode += 1
            scores.append(score)
            score = 0

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                        f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.2f}")

                if train_mode and episode % save_interval == 0:
                    agent.save_model()
