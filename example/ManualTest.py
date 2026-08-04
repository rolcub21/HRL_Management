import random
from example.small_rooms_env import SmallRoomsEnv
from example.Options.pickupOption import PickupOption
from example.Options.storeOption import StoreOption
from example.Options.DeliverOption import DeliverOption
from example.Options.selector import StorageSelectOption
from example.Options.PickupRipeOption import PickupRipeOption
from example.Options.waitOption import WaitOption
from primitive_option import PrimitiveOption


def main():
    env = SmallRoomsEnv(choose_storage=True)

    # Build optionsf
    all_options = []
    for a in env.get_action_space():
        all_options.append(PrimitiveOption(a, env))
    all_options += [
        PickupOption(env),
        StoreOption(env),
        DeliverOption(env),
        WaitOption(env),
        PickupRipeOption(env),  # This option picks up ripe blocks that are ready to be delivered
        #StorageSelectOption(env)
    ]
    for opt in all_options:
        env.options.add(opt)

    state = env.reset()
    current_option = None

    print("Interactive step-by-step: press 'o' to choose option, Enter to step, 'r' to reset, 'q' to quit.")
    while True:
        cmd = input("[o]ption / [Enter] step / [r]eset / [q]uit: ").strip().lower()
        if cmd == 'q':
            print("Exiting.")
            break
        if cmd == 'r':
            state = env.reset()
            current_option = None
            print("Environment reset.")
            env.render()
            continue
        if cmd == 'o' or current_option is None or not current_option.initiation(state):
            # select a new option
            avail = [opt for opt in env.get_available_options(state)
                if opt.initiation(state)]

            print("Available options:")
            for opt in avail:
                print(f" [{all_options.index(opt)}] {opt}")
            choice = input("Choose option index> ").strip()
            try:
                idx = int(choice)
                opt = all_options[idx]
                if not opt.initiation(state):
                    print(f"✗ {opt} can't start here.")
                    continue
                current_option = opt
                print(f"Selected {opt}.")
            except Exception:
                print("Invalid selection.")
            continue
        # step the current option
        print(">>> CURRENT STATE:", state)
        print(">>> CURRENT OPTION:", current_option)
        print(">>> INITIATION OK?:", current_option.initiation(state))
        print(">>> BLOCK POSITIONS:", [b.position for b in env.blocks])
        print(">>> BLOCK STORAGE TIMES (needed/elap):", 
      [(b.label, b.storage_steps_needed, b.stored_time_step, b.storage_steps_elapsed) 
       for b in env.blocks])
        action = current_option.policy(state)
        print(">>> CHOSEN PRIMITIVE:", env.ACTION_NAMES[action])
        nxt, reward, done, info = env.step(action)
        print(f"{current_option} -> {env.ACTION_NAMES[action]}, reward={reward:.2f}, info={info}")
        env.render()
        state = nxt
        if done:
            print("Episode finished.")
            break
        if current_option.termination(state):
            print(f"{current_option} terminated.")
            current_option = None
    
if __name__ == '__main__':
    main()
    main()
