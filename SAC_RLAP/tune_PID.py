import pandas as pd

from SAC_RLAP.evaluate import evaluate

from simglucose.controller.pid_ctrller import PIDController


def tune(args):
    pid_parameters = pd.DataFrame(columns=['Name', 'P', 'I', 'D'])

    for patient_name in args.training_patients:
        score_grid = []
        params_grid = []
        params = (patient_name, 1e-4, 1e-7, 1e-2)

        for _ in range(5):
            _, p, i, d = params
            p_grid = [p/10, p, p*10]
            i_grid = [i/10, i, i*10]
            d_grid = [d/10, d, d*10]

            for p in p_grid:
                for i in i_grid:
                    for d in d_grid:
                        agent = PIDController(P=p, I=i, D=d, target=112)
                        new_score = evaluate(args, agent, [patient_name])
                        new_params = (patient_name, p, i, d)
                        score_grid.append(new_score)
                        params_grid.append(new_params)

            best_score = min(score_grid)
            best_params = params_grid[score_grid.index(best_score)]
            print(f'{patient_name} best parameter {best_params} with score {best_score}')

            if best_params == params: break
            else: params = best_params

        pid_parameters = pid_parameters.append(pd.Series(params, index=['Name', 'P', 'I', 'D']), ignore_index=True)
    
    pid_parameters.sort_values(by=['Name'], inplace=True)
    pid_parameters.to_csv('./SAC_RLAP/pid_parameters.csv', index=False)  