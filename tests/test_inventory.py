"""Unit tests placeholder for Inventory service."""

from backend.services.inventory import InventoryService

def test_inventory_service_initialization():
    """Verify inventory service instance creation."""
    service = InventoryService(dataset_path="Copy_of_sample_cars_dataset.xlsx")
    assert service.dataset_path == "Copy_of_sample_cars_dataset.xlsx"
