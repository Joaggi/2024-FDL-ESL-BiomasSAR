# import matplotlib.pyplot as plt 

# def plot_and_save_comparison(ground_truth, full_data, cropped_data, save_path):
#     # Create a figure with three subplots
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

#     middle_x = full_data.shape[1] // 2
#     x=middle_x
#     y=600
#     # Plot ground truth
#     im1 = ax1.imshow(ground_truth, cmap='viridis', origin='lower')
#     ax1.set_title('Ground Truth')
#     plt.colorbar(im1, ax=ax1)
#     ax1.axvline(x=x, color='r', linestyle='--')
#     ax1.axhline(y=y, color='r', linestyle='--')

#     # Plot full_data (middle slice of z-axis)
#     middle_z = full_data.shape[2] // 2
#     im2 = ax2.imshow(full_data[:, :, middle_z], cmap='viridis', origin='lower')
#     ax2.set_title(f'Full Data (z={middle_z} slice)')
#     plt.colorbar(im2, ax=ax2)
#     ax2.axvline(x=x, color='r', linestyle='--')
#     ax2.axhline(y=y, color='r', linestyle='--')

#     # Plot cropped data (middle slice of z-axis)
#     im3 = ax3.imshow(cropped_data[:, :, middle_z], cmap='viridis', origin='lower')
#     ax3.set_title(f'Cropped Data (z={middle_z} slice)')
#     plt.colorbar(im3, ax=ax3)
#     ax3.axvline(x=x, color='r', linestyle='--')
#     ax3.axhline(y=y, color='r', linestyle='--')

#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     print(f"Figure saved to {save_path}")




# def visualize_quadrant(q_data, q_truth, quadrant_name='Q1'):
#     # Create a figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     # Plot the middle slice of the quadrant data
#     middle_z = q_data.shape[2] // 2
#     im1 = ax1.imshow(q_data[:, :, middle_z], cmap='viridis')
#     ax1.set_title(f'{quadrant_name} Data (z={middle_z} slice)')
#     plt.colorbar(im1, ax=ax1)

#     # Plot the quadrant ground truth
#     im2 = ax2.imshow(q_truth, cmap='viridis')
#     ax2.set_title(f'{quadrant_name} Ground Truth')
#     plt.colorbar(im2, ax=ax2)

#     # Adjust layout and show
#     plt.tight_layout()
#     plt.show()


# def plot_and_save_all_quadrants(quadrants, save_path):
#     # Create a figure with 4 rows (one for each quadrant) and 2 columns (data and ground truth)
#     fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    
#     for i, (quadrant_name, (q_data, q_truth)) in enumerate(quadrants.items()):
#         # Plot the middle slice of the quadrant data
#         middle_z = q_data.shape[2] // 2
#         im1 = axes[i, 0].imshow(q_data[:, :, middle_z], cmap='viridis')
#         axes[i, 0].set_title(f'{quadrant_name.upper()} Data (z={middle_z} slice)')
#         plt.colorbar(im1, ax=axes[i, 0])

#         # Plot the quadrant ground truth
#         im2 = axes[i, 1].imshow(q_truth, cmap='viridis')
#         axes[i, 1].set_title(f'{quadrant_name.upper()} Ground Truth')
#         plt.colorbar(im2, ax=axes[i, 1])

#         # Remove axes ticks for cleaner look
#         axes[i, 0].set_xticks([])
#         axes[i, 0].set_yticks([])
#         axes[i, 1].set_xticks([])
#         axes[i, 1].set_yticks([])

#     # Adjust layout and save
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"Figure saved to {save_path}")


