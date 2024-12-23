package com.model;

public class SubServiceModel {
private int id,parentid;
private String title,description,cost;
public int getId() {
	return id;
}
public void setId(int id) {
	this.id = id;
}
public int getParentid() {
	return parentid;
}
public void setParentid(int parentid) {
	this.parentid = parentid;
}
public String getTitle() {
	return title;
}
public void setTitle(String title) {
	this.title = title;
}
public String getDescription() {
	return description;
}
public void setDescription(String description) {
	this.description = description;
}
public String getCost() {
	return cost;
}
public void setCost(String cost) {
	this.cost = cost;
}
}
