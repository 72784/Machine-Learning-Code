<!DOCTYPE html>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<html lang="en">

<head>
  <!-- Required meta tags -->
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <title>RoyalUI Admin</title>
  <!-- plugins:css -->
  <link rel="stylesheet" href="vendors/ti-icons/css/themify-icons.css">
  <link rel="stylesheet" href="vendors/base/vendor.bundle.base.css">
  <!-- endinject -->
  <!-- plugin css for this page -->
  <!-- End plugin css for this page -->
  <!-- inject:css -->
  <link rel="stylesheet" href="css/style.css">
  <!-- endinject -->
  <link rel="shortcut icon" href="images/favicon.png" />
</head>
<body>
  <div class="container-scroller">
    <!-- partial:partials/_navbar.html -->
    <nav class="navbar col-lg-12 col-12 p-0 fixed-top d-flex flex-row">
      <div class="text-center navbar-brand-wrapper d-flex align-items-center justify-content-center">
        <a class="navbar-brand brand-logo mr-5" href="index.html"><img src="images/logo.svg" class="mr-2" alt="logo"/></a>
        <a class="navbar-brand brand-logo-mini" href="index.html"><img src="images/logo-mini.svg" alt="logo"/></a>
      </div>
      <div class="navbar-menu-wrapper d-flex align-items-center justify-content-end">
        <button class="navbar-toggler navbar-toggler align-self-center" type="button" data-toggle="minimize">
          <span class="ti-view-list"></span>
        </button>
        <ul class="navbar-nav mr-lg-2">
          <li class="nav-item nav-search d-none d-lg-block">
            <div class="input-group">
              <div class="input-group-prepend hover-cursor" id="navbar-search-icon">
                <span class="input-group-text" id="search">
                  <i class="ti-search"></i>
                </span>
              </div>
              <input type="text" class="form-control" id="navbar-search-input" placeholder="Search now" aria-label="search" aria-describedby="search">
            </div>
          </li>
        </ul>
        <ul class="navbar-nav navbar-nav-right">
          <li class="nav-item dropdown mr-1">
            <a class="nav-link count-indicator dropdown-toggle d-flex justify-content-center align-items-center" id="messageDropdown" href="#" data-toggle="dropdown">
              <i class="ti-email mx-0"></i>
            </a>
            <div class="dropdown-menu dropdown-menu-right navbar-dropdown" aria-labelledby="messageDropdown">
              <p class="mb-0 font-weight-normal float-left dropdown-header">Messages</p>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                    <img src="images/faces/face4.jpg" alt="image" class="profile-pic">
                </div>
                <div class="item-content flex-grow">
                  <h6 class="ellipsis font-weight-normal">David Grey
                  </h6>
                  <p class="font-weight-light small-text text-muted mb-0">
                    The meeting is cancelled
                  </p>
                </div>
              </a>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                    <img src="images/faces/face2.jpg" alt="image" class="profile-pic">
                </div>
                <div class="item-content flex-grow">
                  <h6 class="ellipsis font-weight-normal">Tim Cook
                  </h6>
                  <p class="font-weight-light small-text text-muted mb-0">
                    New product launch
                  </p>
                </div>
              </a>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                    <img src="images/faces/face3.jpg" alt="image" class="profile-pic">
                </div>
                <div class="item-content flex-grow">
                  <h6 class="ellipsis font-weight-normal"> Johnson
                  </h6>
                  <p class="font-weight-light small-text text-muted mb-0">
                    Upcoming board meeting
                  </p>
                </div>
              </a>
            </div>
          </li>
          <li class="nav-item dropdown">
            <a class="nav-link count-indicator dropdown-toggle" id="notificationDropdown" href="#" data-toggle="dropdown">
              <i class="ti-bell mx-0"></i>
              <span class="count"></span>
            </a>
            <div class="dropdown-menu dropdown-menu-right navbar-dropdown" aria-labelledby="notificationDropdown">
              <p class="mb-0 font-weight-normal float-left dropdown-header">Notifications</p>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                  <div class="item-icon bg-success">
                    <i class="ti-info-alt mx-0"></i>
                  </div>
                </div>
                <div class="item-content">
                  <h6 class="font-weight-normal">Application Error</h6>
                  <p class="font-weight-light small-text mb-0 text-muted">
                    Just now
                  </p>
                </div>
              </a>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                  <div class="item-icon bg-warning">
                    <i class="ti-settings mx-0"></i>
                  </div>
                </div>
                <div class="item-content">
                  <h6 class="font-weight-normal">Settings</h6>
                  <p class="font-weight-light small-text mb-0 text-muted">
                    Private message
                  </p>
                </div>
              </a>
              <a class="dropdown-item">
                <div class="item-thumbnail">
                  <div class="item-icon bg-info">
                    <i class="ti-user mx-0"></i>
                  </div>
                </div>
                <div class="item-content">
                  <h6 class="font-weight-normal">New user registration</h6>
                  <p class="font-weight-light small-text mb-0 text-muted">
                    2 days ago
                  </p>
                </div>
              </a>
            </div>
          </li>
          <li class="nav-item nav-profile dropdown">
            <a class="nav-link dropdown-toggle" href="#" data-toggle="dropdown" id="profileDropdown">
              <img src="images/faces/face28.jpg" alt="profile"/>
            </a>
            <div class="dropdown-menu dropdown-menu-right navbar-dropdown" aria-labelledby="profileDropdown">
              <a class="dropdown-item">
                <i class="ti-settings text-primary"></i>
                Settings
              </a>
              <a class="dropdown-item">
                <i class="ti-power-off text-primary"></i>
                Logout
              </a>
            </div>
          </li>
        </ul>
        <button class="navbar-toggler navbar-toggler-right d-lg-none align-self-center" type="button" data-toggle="offcanvas">
          <span class="ti-view-list"></span>
        </button>
      </div>
    </nav>
    
    <jsp:include page="sidebar.jsp" />
    
          <div class="col-lg-12 grid-margin stretch-card">
              <div class="card">
                <div class="card-body">
                  <h4 class="card-title">Bordered table</h4>
                  <p class="card-description">
                    Add class <code>.table-bordered</code>
                  </p>
                  <div class="table-responsive pt-3">
                    <table class="table table-bordered">
                      <thead>
                        <tr>
                          <th>
                            Book id
                          </th>
                          <th>
                            Customer Name
                          </th>
                          <th>
                            City
                          </th>
                          <th>
                           Booking Date
                          </th>
                          <th>
                            Sub-Service Requirement
                          </th>
                          <th>
                            Description
                          </th>
                          <th>
                            Price
                          </th>
                          <th>
                            Status
                          </th>
                           <th style="padding-left: 110px" >
                            Assignment
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                                <%
									Connection cn3 = new DBUtil().getDatabaseConnection();
									PreparedStatement stfirst = cn3.prepareStatement("select * from assign where servicemen=?");
									stfirst.setString(1, request.getParameter("servicemen"));
									ResultSet rs3 = stfirst.executeQuery();
									while (rs3.next()) {
									%>              
                      
                       <%
                                          Connection cn=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st = cn.prepareStatement("select * from book where bookid=?");
                                             st.setString(1, rs3.getString("bookid"));
                                         ResultSet rs=st.executeQuery();
                                         while(rs.next())
                                         {
                                      %>
                        <tr>
                         <%
                                             Connection cn1=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st1 = cn1.prepareStatement("select * from customer where userid=?");
                                             st1.setString(1, rs.getString("userid"));
                                         ResultSet rs1=st1.executeQuery();
                                         while(rs1.next())
                                         {
                                      %>
                          <td>
                            <%= rs.getString("bookid")%>;
                          </td>
                              <td>
                              <%=rs1.getString("firstname") +" "+rs1.getString("lastname") %>
                              </td>

                          <td>
                            <%= rs1.getString("city")%>
                          </td>
                          <%
                                         }
                                         cn1.close();
				                     %>
				                     
                          <td>
                             <%= rs.getString("bdate")%>
                          </td>
                           <%
                                             Connection cn2=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st2 = cn2.prepareStatement("select * from subservicei where id=?");
                                             st2.setString(1, rs.getString("id"));
                                         ResultSet rs2=st2.executeQuery();
                                         while(rs2.next())
                                         {
                                             %>
                                                     <td>
                                                       <%=rs2.getString("title") %>
                                                    </td>
                                                    <td>
                                                       <%=rs2.getString("description") %>
                                                    </td>
                                                    
                                                <%
                                         }
                                         cn2.close();
				                     %>
				                      <td>
                             <%= rs.getString("price")%>
                          </td>
				                      <td>
                             <%= rs.getString("status")%>
                          </td>
                       
                        
                        
						<td style=" ">
										<%if (rs.getString("status").equalsIgnoreCase("Contract Accepted")){ %>
										<a href="view_contract_details.jsp?bookid=<%= rs3.getString("bookid") %>"> <button class="btn btn-sm btn-outline-dark">View Customer Details</button></a></td>
										<%}else if(rs.getString("status").equalsIgnoreCase("Contract Closed")){ %>				
                        Contract Succefully completed 
                        <a
									href=""><button
										type="submit" name="action" value="contract"
										class="btn btn-outline-info">Feedback Form</button></a>
								</td>
              <%}else{ %>          
  <a href="contract_details.jsp?bookid=<%=rs3.getString("bookid")%>"><button
													class="btn btn-sm btn-outline-primary">Accept</button></a>
													<form style="display: inline" action="../AdminController" method="post">
											<input type="hidden" name="bookid" value="<%=rs3.getString("bookid")%>">
											<button type="submit" name="action" value="requestdenied" class="btn btn-sm btn-outline-danger">Decline</button>
											</form>
                                                 </td>
                       
                         										<%} %>
				                                <%
                                         }
                                         cn.close();
				                     %>
				                 <%
                                         }
                                         cn3.close();
				                     %>     
                        
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
                  <!-- partial:partials/_footer.html -->
        <footer class="footer">
          <div class="d-sm-flex justify-content-center justify-content-sm-between">
            <span class="text-muted text-center text-sm-left d-block d-sm-inline-block">Copyright © 2018 <a href="https://www.templatewatch.com/" target="_blank">Templatewatch</a>. All rights reserved.</span>
            <span class="float-none float-sm-right d-block mt-1 mt-sm-0 text-center">Hand-crafted & made with <i class="ti-heart text-danger ml-1"></i></span>
          </div>
        </footer>
        <!-- partial -->
      </div>
      <!-- main-panel ends -->
    </div>
    <!-- page-body-wrapper ends -->
  </div>
  <!-- container-scroller -->

  <!-- plugins:js -->
  <script src="vendors/base/vendor.bundle.base.js"></script>
  <!-- endinject -->
  <!-- Plugin js for this page-->
  <script src="vendors/chart.js/Chart.min.js"></script>
  <!-- End plugin js for this page-->
  <!-- inject:js -->
  <script src="js/off-canvas.js"></script>
  <script src="js/hoverable-collapse.js"></script>
  <script src="js/template.js"></script>
  <script src="js/todolist.js"></script>
  <!-- endinject -->
  <!-- Custom js for this page-->
  <script src="js/dashboard.js"></script>
  <!-- End custom js for this page-->
</body>

</html>

