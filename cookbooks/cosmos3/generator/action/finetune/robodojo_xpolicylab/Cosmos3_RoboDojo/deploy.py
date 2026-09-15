def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action_index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        observations = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=observations)
        action_chunks = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        for action_index in range(len(action_chunks[0])):
            actions = [chunk[action_index] for chunk in action_chunks]
            TASK_ENV.take_action_batch(actions, env_idx_list)
            if TASK_ENV.is_episode_end() or action_index + 1 == len(action_chunks[0]):
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_indexes = [
                index for index, env_idx in enumerate(env_idx_list) if env_idx in running
            ]
            action_chunks = [action_chunks[index] for index in active_batch_indexes]
            env_idx_list = [env_idx_list[index] for index in active_batch_indexes]
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_idx_list),
            )