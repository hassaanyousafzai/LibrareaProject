# This dictionary will store the state of each background upload task.
# The key will be the image_id (UUID), and the value will be a dictionary
# containing the task's status, any potential errors, and a cancellation flag.
#
# Example structure:
# {
#   "some-uuid-string": {
#     "status": "processing" | "completed" | "failed" | "cancelled",
#     "cancel": False,
#     "error": "optional error message"
#   }
# }
upload_tasks = {}
