import func_experiment as f

#需要调整的参数P、I、D，此处的默认值为论文代码给出
update_pid_template = {
        "MC_ROLLRATE_P": 0.4,
        "MC_PITCHRATE_P": 0.4,
        "MC_YAWRATE_P": 0.1,
        "MC_ROLLRATE_I": 0.07,
        "MC_PITCHRATE_I": 0.07,
        "MC_YAWRATE_I": 0.0005,
        "MC_ROLLRATE_D": 0.0016,
        "MC_PITCHRATE_D": 0.0016,
        "MC_YAWRATE_D": 0.01
    }

######性能指标ACE
ace_score = f.calculate_ACE(
        update_pid=update_pid_template,
        # update_pid=None,
        need_pdf=True
)

print("ace_score is:", ace_score)