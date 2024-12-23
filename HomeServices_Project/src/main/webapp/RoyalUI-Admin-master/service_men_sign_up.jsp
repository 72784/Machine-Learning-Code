<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<link href="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
<script src="//maxcdn.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
<script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>

<link href="register_styles.css" rel="stylesheet">

<meta charset="ISO-8859-1">
<title>Insert title here</title>
</head>
<body>

<div class="container register" style="height:535px">
                <div class="row">
                    <div class="col-md-3 register-left">
                        <img src="https://image.ibb.co/n7oTvU/logo_white.png" alt=""/>
                        <h3>Welcome</h3>
                        <p>You are 30 seconds away from earning your own money!</p>
                        <input style="padding:12px; margin-top: -24px" type="submit" name="" value="Login"/><br/>
                    </div>
                    
<form action="../AdminController" method="post">
                    
                    <div class="col-md-15 register-right">
                        <div class="tab-content" id="myTabContent">
                            <div class="tab-pane fade show active" id="home" role="tabpanel" aria-labelledby="home-tab">
                                <h3 class="register-heading" style="padding:14px">Apply as a Employee</h3>
                                <div class="row register-form" >
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <input type="text" name="firstname" class="form-control" placeholder="First Name *" value="" />
                                        </div>
                                        <div class="form-group">
                                            <input type="text" name="lastname" class="form-control" placeholder="Last Name *" value="" />
                                        </div>
                                        <div class="form-group">
                                            <input type="password" name="password" class="form-control" placeholder="Password *" value="" />
                                        </div>
                                        
                                        <div class="form-group">
                                            <div class="maxl">
                                                <label class="radio inline"> 
                                                    <input type="radio" name="gender" value="male" checked>
                                                    <span> Male </span> 
                                                </label>
                                                <label class="radio inline"> 
                                                    <input type="radio" name="gender" value="female">
                                                    <span>Female </span> 
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <input type="email" name="email" class="form-control" placeholder="Your Email *" value="" />
                                        </div>
                                        <div class="form-group">
                                            <input type="text" name="phone" minlength="10" maxlength="10"  class="form-control" placeholder="Your Phone *" value="" />
                                        </div>
                                        <div class="form-group">
                                            <select class="form-control" name="specialization">
                                                <option class="hidden" selected="" disabled="">Select Service</option>
                                                 <%
                                             Connection cn=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st = cn.prepareStatement("select * from mainservice");
                                         ResultSet rs=st.executeQuery();
                                         while(rs.next())
                                         {
                                             %>
                                                <option  value="<%= rs.getString(2) %>"><%= rs.getString(2) %></option>
                                                <%
                                         }
                                         cn.close();
				%>
                                            </select>
                                      
                                    </div>
                                      
                                        <button type="submit" name="action" value="employee" class="btnRegister"  value="Register"/>Sign Up </button>
                                    </div>
                                </div>
                            </div>
                            
                        </div>
                    </div>
                </div>
</form>

</body>
</html>