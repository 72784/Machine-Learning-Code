package com.model;

public class ServiceModel {
private int serviceid;
private String type,description, image;
public String getImage() {
	return image;
}
public void setImage(String image) {
	this.image = image;
}
public int getServiceid() {
	return serviceid;
}
public void setServiceid(int serviceid) {
	this.serviceid = serviceid;
}
public String getType() {
	return type;
}
public void setType(String type) {
	this.type = type;
}
public String getDescription() {
	return description;
}
public void setDescription(String description) {
	this.description = description;
}
}
