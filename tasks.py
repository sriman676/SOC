
from graders import (
	grade_credential_access_detection,
	grade_data_exfiltration_detection,
	grade_login_attack_detection,
	grade_normal_activity_handling,
	grade_privilege_escalation_detection,
)


TASKS = {
	"login_attack_detection": {
		"event_id": 4625,
		"stage": "initial_access",
		"weight": 0.18,
		"grader": grade_login_attack_detection,
		"grader_name": "graders.grade_login_attack_detection",
	},
	"normal_activity_handling": {
		"event_id": 4624,
		"stage": "normal_activity",
		"weight": 0.12,
		"grader": grade_normal_activity_handling,
		"grader_name": "graders.grade_normal_activity_handling",
	},
	"privilege_escalation_detection": {
		"event_id": 4672,
		"stage": "privilege_escalation",
		"weight": 0.25,
		"grader": grade_privilege_escalation_detection,
		"grader_name": "graders.grade_privilege_escalation_detection",
	},
	"credential_access_detection": {
		"event_id": 4688,
		"stage": "credential_access",
		"weight": 0.20,
		"grader": grade_credential_access_detection,
		"grader_name": "graders.grade_credential_access_detection",
	},
	"data_exfiltration_detection": {
		"event_id": 5156,
		"stage": "exfiltration",
		"weight": 0.25,
		"grader": grade_data_exfiltration_detection,
		"grader_name": "graders.grade_data_exfiltration_detection",
	},
}
