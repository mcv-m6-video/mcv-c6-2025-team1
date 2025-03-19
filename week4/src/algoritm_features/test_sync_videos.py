import cv2
import numpy as np
import math


def synchronize_videos(video_paths, delays, output_path="synchronized_output_s04_3.avi", target_fps=10):
	"""
	Synchronize multiple videos based on their starting delays.

	Args:
		video_paths (list): List of paths to input videos.
		delays (list): List of delays (in seconds) for each video.
		output_path (str): Path to save the synchronized output video.
		target_fps (int): Target frame rate for the output video.
	"""
	# Open all video captures
	caps = [cv2.VideoCapture(path) for path in video_paths]

	# Get video properties
	frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
	fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
	print(fps_values)
	widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
	heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

	# Calculate frame delays
	frame_delays = [int(delay * fps) for delay, fps in zip(delays, fps_values)]

	# Calculate total frames needed for output
	adjusted_frame_counts = [count + delay for count, delay in zip(frame_counts, frame_delays)]
	total_frames = max(adjusted_frame_counts)

	# Calculate output video dimensions
	# Assuming we'll create a grid layout
	#rows = 2
	#cols = 3
	cols = math.ceil(math.sqrt(len(video_paths)))
	rows = math.ceil(len(video_paths) / cols)
	output_width = max(widths) * cols
	output_height = max(heights) * rows

	# Create video writer
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(output_path, fourcc, target_fps, (output_width, output_height))

	# Process frames
	for frame_idx in range(total_frames):
		# Create a blank canvas for the current frame
		output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

		# Process each video
		for i, (cap, delay, fps) in enumerate(zip(caps, frame_delays, fps_values)):
			# Calculate the source frame index, accounting for frame rate differences
			source_frame_idx = frame_idx - delay

			# If frame rate is different from target, adjust the source frame index
			if fps != target_fps:
				source_frame_idx = int(source_frame_idx * (fps / target_fps))

			# Skip if we're before the start of this video
			if source_frame_idx < 0:
				continue

			# Skip if we're past the end of this video
			if source_frame_idx >= frame_counts[i]:
				continue

			# Set the frame position and read the frame
			cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
			ret, frame = cap.read()

			if not ret:
				continue

			# Resize frame if necessary
			frame = cv2.resize(frame, (widths[0], heights[0]))

			# Calculate position in grid
			row = i // cols
			col = i % cols
			y_offset = row * heights[0]
			x_offset = col * widths[0]

			# Place the frame in the output canvas
			output_frame[y_offset:y_offset + heights[0], x_offset:x_offset + widths[0]] = frame

		# Write the combined frame to output video
		out.write(output_frame)

		# Display progress
		if frame_idx % 100 == 0:
			print(f"Processing frame {frame_idx}/{total_frames}")

	# Release resources
	for cap in caps:
		cap.release()
	out.release()

	print(f"Synchronized video saved as {output_path}")


def main():
	# Example usage
	video_paths = [
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c010.avi',
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c011.avi',
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c012.avi',
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c013.avi',
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c014.avi',
		'/ghome/c5mcv01/mcv-c6-2025-team1/week4/src/approach2_c015.avi'
	]

	# Delays in seconds from the provided data
	delays = [8.715, 8.457, 5.879, 0, 5.042, 8.492]
	#delays = [0, 1.640, 2.049, 2.177, 2.235]
	#delays = [0, 14.318, 29.955, 26.979, 25.905, 39.973, 49.422, 45.716]
	#delays = [125.199, 150.893, 140.218, 165.568, 170.797, 170.567, 175.426, 175.644, 175.838]

	# Call the synchronization function
	synchronize_videos(video_paths, delays, output_path="synchronized_output_s03_ap2.avi")


if __name__ == "__main__":
	main()