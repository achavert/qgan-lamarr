from .dashboard import TrainingDashboard

run_tag = str(input('run ID:'))
monitor = TrainingDashboard("./output/"+run_tag, refresh_rate = 0.5)
monitor.run(port = 8055)